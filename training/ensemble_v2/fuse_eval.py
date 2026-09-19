"""Fuse the ensemble members, refine outlines, and score every stage on the same
held-out areas (Gandhinagar test strip, RAMP test strips).

Members (predictions written by their own kernels, plus one run here):
    unet     U-Net multimask (RAMP targets, Nacala init) -- also gives contact/interior maps
    yolo     YOLOv8-seg (Nacala init)
    mrcnn    Mask R-CNN trained on RAMP + Gandhinagar
    mrcnn_gn Mask R-CNN trained on Gandhinagar only (the earlier fair-split model)

Stages, each scored separately so we can see what every step buys:
    1. vote     instances matched across members by IoU; a house is kept when at
                least VOTES_MIN members agree (or one member is very confident);
                its mask is the pixel-wise majority of the members that found it
    2. split    a fused house that holds two U-Net interior seeds separated by the
                predicted contact band is cut in two along that band
    3. sam      SAM (ViT-B) is prompted with each house's box + centre point; its
                outline replaces ours only when the two agree (IoU >= SAM_ACCEPT)
    4. regular  outlines squared up with buildingregulariser (ArcGIS-style
                regularisation, neighbour alignment on)
Thresholds are fixed up front, not tuned on the test areas.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segment-anything", "buildingregulariser"], check=True)
import torch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from common import (MIN_HOUSE_PX, find, load_gandhinagar, load_ramp, log, masks_to_labels, pack_instances,  # noqa: E402
                    panel, relabel, score_tile, summarise, unpack_instances)

WORK = Path("/kaggle/working")
DEV = "cuda"
MATCH_IOU = 0.5
VOTES_MIN = 2
SOLO_SCORE = 0.85
SAM_ACCEPT = 0.65


# ------------------------------------------------------------------ inputs
def load_member(fname):
    f = find(fname)
    if not f:
        log("missing", fname); return None
    return np.load(f[0], allow_pickle=True).item()


def run_mrcnn_gn(tests):
    """Earlier Gandhinagar-only Mask R-CNN (fair split), same scale it was trained at."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    f = find("maskrcnn_rgb.pt")
    if not f:
        return None
    m = maskrcnn_resnet50_fpn_v2(weights=None, box_detections_per_img=400)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    m.load_state_dict(torch.load(f[0], map_location="cpu")); m.to(DEV).eval()
    out = {}
    with torch.no_grad():
        for name, samples in tests.items():
            recs = {}
            for sid, rgb, _, _ in samples:
                x = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)).to(DEV)
                # trained on 1024 Gandhinagar tiles cropped to 512 at native scale
                m.transform.min_size, m.transform.max_size = (rgb.shape[0],), rgb.shape[0]
                o = m([x])[0]; k = o["scores"] >= 0.3
                recs[sid] = pack_instances(list((o["masks"][k][:, 0] > 0.5).cpu().numpy()), o["scores"][k].cpu().numpy())
            out[name] = recs
    del m; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ stage 1: vote
def iou(a, b):
    i = np.logical_and(a, b).sum()
    return i / max(1, np.logical_or(a, b).sum())


def vote(member_preds, shape, n_members):
    """member_preds: {member: (masks, scores)} -> fused masks, scores, votes."""
    items = [(s, mem, m) for mem, (ms, ss) in member_preds.items() for m, s in zip(ms, ss)]
    items.sort(key=lambda t: -t[0])
    clusters = []   # each: {"members": {mem: (mask, score)}, "rep": mask}
    for s, mem, m in items:
        best, bi = 0, -1
        for ci, c in enumerate(clusters):
            if mem in c["members"]:
                continue
            v = iou(m, c["rep"])
            if v > best:
                best, bi = v, ci
        if best >= MATCH_IOU:
            clusters[bi]["members"][mem] = (m, s)
        else:
            clusters.append({"members": {mem: (m, s)}, "rep": m})
    masks, scores, votes = [], [], []
    for c in clusters:
        k = len(c["members"])
        sc = float(np.mean([s for _, s in c["members"].values()]))
        if k < VOTES_MIN and sc < SOLO_SCORE:
            continue
        stack = np.stack([m for m, _ in c["members"].values()])
        fused = stack.mean(0) >= 0.5 if k > 1 else stack[0]
        if fused.sum() < MIN_HOUSE_PX:
            continue
        masks.append(fused); scores.append(sc * min(1.0, k / min(3, n_members))); votes.append(k)
    return masks, np.array(scores, np.float32), votes


# ------------------------------------------------------------------ stage 2: contact split
def contact_split(masks, scores, maps):
    """maps: uint8 (4,H,W) roof, contact, interior, distance from the U-Net (native scale)."""
    if maps is None:
        return masks, scores
    from skimage.segmentation import watershed
    contact = maps[1] / 255.0; inter = maps[2] / 255.0; dist = maps[3] / 255.0
    seeds_all = (inter > 0.5) & (contact < 0.35)
    out_m, out_s = [], []
    for m, s in zip(masks, scores):
        lab, n = ndi.label(seeds_all & m)
        if n >= 2:
            sizes = np.bincount(lab.ravel())[1:]
            big = np.nonzero(sizes >= 6)[0] + 1
            if len(big) >= 2:
                mk = np.where(np.isin(lab, big), lab, 0)
                parts = watershed(-dist + contact, markers=mk, mask=m)
                pieces = [parts == v for v in big if (parts == v).sum() >= MIN_HOUSE_PX]
                if len(pieces) >= 2:
                    out_m += pieces; out_s += [s] * len(pieces)
                    continue
        out_m.append(m); out_s.append(s)
    return out_m, np.array(out_s, np.float32)


# ------------------------------------------------------------------ stage 3: SAM
_sam = None


def sam_predictor():
    global _sam
    if _sam is None:
        from segment_anything import SamPredictor, sam_model_registry
        ck = Path("/kaggle/tmp/sam_vit_b_01ec64.pth")
        if not ck.exists():
            ck.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", ck)
        _sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(ck)).to(DEV))
    return _sam


@torch.no_grad()
def sam_refine(rgb, masks):
    if not masks:
        return masks, 0
    p = sam_predictor(); p.set_image(rgb)
    boxes, pts = [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min() - 1, ys.min() - 1, xs.max() + 1, ys.max() + 1])
        d = ndi.distance_transform_edt(m); cy, cx = np.unravel_index(np.argmax(d), d.shape)
        pts.append([[cx, cy]])
    tb = p.transform.apply_boxes_torch(torch.tensor(boxes, dtype=torch.float32, device=DEV), rgb.shape[:2])
    tp = p.transform.apply_coords_torch(torch.tensor(pts, dtype=torch.float32, device=DEV), rgb.shape[:2])
    out, used = [], 0
    for i in range(0, len(masks), 64):
        sm, sc, _ = p.predict_torch(point_coords=tp[i:i + 64], point_labels=torch.ones(tp[i:i + 64].shape[:2], device=DEV),
                                    boxes=tb[i:i + 64], multimask_output=False)
        for j, s in enumerate(sm[:, 0].cpu().numpy()):
            m = masks[i + j]
            if iou(s, m) >= SAM_ACCEPT and 0.7 <= s.sum() / max(1, m.sum()) <= 1.4:
                out.append(s); used += 1
            else:
                out.append(m)
    return out, used


# ------------------------------------------------------------------ stage 4: regularise
def regularise(lab):
    import geopandas as gpd
    import rasterio.features
    from buildingregulariser import regularize_geodataframe
    from shapely.geometry import shape
    geoms, ids = [], []
    for g, v in rasterio.features.shapes(lab.astype(np.int32), mask=lab > 0):
        geoms.append(shape(g)); ids.append(int(v))
    if not geoms:
        return lab
    # pixel units in a metric CRS; tolerances are in pixels (1 px ~ 0.3 m)
    gdf = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:3857")
    gdf = gdf.dissolve("id").reset_index()
    try:
        reg = regularize_geodataframe(gdf, simplify_tolerance=1.0, parallel_threshold=1.5, allow_45_degree=True,
                                      neighbor_alignment=True, neighbor_search_distance=40, neighbor_max_rotation=10,
                                      num_cores=1)
    except Exception as e:
        log("regulariser failed:", repr(e)[:200]); return lab
    out = np.zeros(lab.shape, np.int32)
    order = np.argsort(-reg.geometry.area.values)   # small houses painted last keep their pixels
    shapes = [(reg.geometry.values[i], int(reg["id"].values[i])) for i in order if not reg.geometry.values[i].is_empty]
    if shapes:
        out = rasterio.features.rasterize(shapes, out_shape=lab.shape, dtype="int32")
    return relabel(out)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    tests = {"gandhinagar": load_gandhinagar("test"), "ramp": load_ramp("test")}
    preds = {"unet": load_member("pred_unet.npy"), "yolo": load_member("pred_yolo.npy"),
             "mrcnn": load_member("pred_maskrcnn.npy")}
    maps = load_member("maps_unet.npy")
    preds["mrcnn_gn"] = run_mrcnn_gn(tests)
    members = [k for k, v in preds.items() if v is not None]
    log("members:", members)

    stages = members + ["vote", "vote+split", "vote+split+sam", "vote+split+sam+regular"]
    results = {}
    for name, samples in tests.items():
        rows = {s: [] for s in stages}
        pics = []
        for n, (sid, rgb, gt, partial) in enumerate(samples):
            gt = gt.astype(np.int32); shape = gt.shape
            per = {mem: unpack_instances(preds[mem][name].get(sid, []), shape) for mem in members}
            labs = {}
            for mem in members:
                ms, ss = per[mem]
                labs[mem] = masks_to_labels(ms, ss) if ms else np.zeros(shape, np.int32)
            fm, fs, _ = vote(per, shape, len(members))
            labs["vote"] = masks_to_labels(fm, fs) if fm else np.zeros(shape, np.int32)
            mp = maps[name].get(sid) if maps else None
            sm, ss = contact_split(fm, fs, mp)
            labs["vote+split"] = masks_to_labels(sm, ss) if sm else np.zeros(shape, np.int32)
            rm, _ = sam_refine(rgb, sm)
            labs["vote+split+sam"] = masks_to_labels(rm, ss) if rm else np.zeros(shape, np.int32)
            labs["vote+split+sam+regular"] = regularise(labs["vote+split+sam"])
            for s in stages:
                rows[s].append(score_tile(labs[s], gt, partial))
            if len(pics) < (6 if name == "gandhinagar" else 4):
                pics.append((rgb, [None, gt] + [labs[m] for m in members] + [labs["vote+split"], labs["vote+split+sam+regular"]]))
            if n % 200 == 0:
                log(name, n, "/", len(samples))
        results[name] = {s: summarise(rows[s]) for s in stages}
        panel(pics, ["image", "label"] + members + ["vote+split", "final (+SAM+regular)"],
              WORK / f"ensemble_{name}.jpg", scale=0.8 if name == "ramp" else 1)
        log(name, json.dumps(results[name], indent=1))
    (WORK / "ensemble_results.json").write_text(json.dumps(results, indent=2))
    log("done in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
