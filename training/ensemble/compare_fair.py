"""Stacking comparison on the FAIR Gandhinagar split (Kaggle T4): tiles cut from the
rebuilt photo, learning side west of the line, exam side east of it, so no house is
in both. Does feeding the U-Net's output into an instance-segmentation CNN beat each
model on its own? Edge-case training is on for all: seam weighting for the U-Net,
crowded-row sampling for Mask R-CNN.

    A  U-Net + roof edges (fine-tuned from the WHU model), roofs split by seeds + edges
    B  Mask R-CNN on the image alone
    D  Mask R-CNN on image + A's roof / edge / seed maps (6 input channels)

A is trained twice on two halves of the training tiles, so every training tile gets
an A-prediction from a model that never saw it (out-of-fold). D learns from those,
so it learns when to trust A and when to override it. Test tiles get the average of
both A models. All three are scored the same way on the de-duplicated test split.

Expects /kaggle/working/train.py and /kaggle/working/prepare_gandhinagar.py.
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi

sys.path.insert(0, "/kaggle/working")
import train as tr  # noqa: E402  (the dense-instance trainer, for its decoder and constants)

import segmentation_models_pytorch as smp  # noqa: E402
import torchvision  # noqa: E402
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor  # noqa: E402

DEV = "cuda"
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp/pilot")
S1_EPOCHS, S2_STEPS = int(os.environ.get("S1_EPOCHS", 60)), int(os.environ.get("S2_STEPS", 1500))
CROP = 512
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ scoring
def score_tile(pred, gt):
    """IoU matrix from one contingency table: houses found (IoU>=0.5), outline IoU of
    found houses, neighbouring house pairs that share one predicted shape."""
    g = gt.ravel().astype(np.int64); p = pred.ravel().astype(np.int64)
    G, P = int(g.max()) + 1, int(p.max()) + 1
    cont = np.bincount(g * P + p, minlength=G * P).reshape(G, P)
    inter = cont[1:, 1:]
    union = cont[1:].sum(1)[:, None] + cont[:, 1:].sum(0)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    best = iou.max(1) if iou.size else np.zeros(G - 1)
    found = best >= 0.5
    extra = int((iou.max(0) < 0.5).sum()) if iou.size else P - 1
    dom = np.where(inter.sum(1) > 0, inter.argmax(1) + 1, 0) if inter.size else np.zeros(G - 1, int)
    BIG = 1 << 30
    gmax = ndi.maximum_filter(gt, 7)
    gmin = ndi.minimum_filter(np.where(gt > 0, gt, BIG), 7)
    ok = (gmin < BIG) & (gmax > 0) & (gmin != gmax)
    pairs = set(zip(gmin[ok].tolist(), gmax[ok].tolist()))
    merged = sum(1 for a, b in pairs if dom[a - 1] and dom[a - 1] == dom[b - 1])
    return dict(total=G - 1, found=int(found.sum()), iou_sum=float(best[found].sum()),
                pairs=len(pairs), merged=merged, extra=extra,
                pix_i=int(((pred > 0) & (gt > 0)).sum()), pix_u=int(((pred > 0) | (gt > 0)).sum()))


def summarise(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    return dict(houses_found=f"{s['found']}/{s['total']}", recall=round(s["found"] / max(1, s["total"]), 3),
                outline_iou=round(s["iou_sum"] / max(1, s["found"]), 3),
                merged_pairs=f"{s['merged']}/{s['pairs']}", merge_rate=round(s["merged"] / max(1, s["pairs"]), 3),
                unmatched_shapes=s["extra"], building_iou=round(s["pix_i"] / max(1, s["pix_u"]), 3))


# ------------------------------------------------------------------ stage 1 (U-Net)
def load_unet(path):
    ck = torch.load(path, map_location="cpu")
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=ck.get("n_out", 8))
    net.load_state_dict(ck["state_dict"]); net.to(DEV).eval()
    return net, np.array(ck["mean"], np.float32), np.array(ck["std"], np.float32)


@torch.no_grad()
def unet_outputs(models, rgb):
    x = np.dstack([rgb.astype(np.float32), np.zeros(rgb.shape[:2], np.float32)]) / 255.0
    outs = []
    for net, mean, std in models:
        t = torch.from_numpy(((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
        outs.append(net(t).float()[0])
    o = torch.stack(outs).mean(0).cpu()
    ps = o[tr.SEM].softmax(0).numpy()
    pe = torch.sigmoid(o[tr.EDGE]).numpy(); pi = torch.sigmoid(o[tr.INT]).numpy()
    pd = o[tr.DIST].clamp(0, 1).numpy()
    return ps, pe, pi, pd


def maps_u8(ps, pe, pi):
    return np.stack([ps[tr.CLASSES.index("building")], pe, pi], -1).__mul__(255).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------ stage 2 (Mask R-CNN)
def build_maskrcnn(in_ch):
    m = maskrcnn_resnet50_fpn_v2(weights="DEFAULT", box_detections_per_img=300,
                                 rpn_pre_nms_top_n_test=3000, rpn_post_nms_top_n_test=2000)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    if in_ch > 3:
        old = m.backbone.body.conv1
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(1, keepdim=True).repeat(1, in_ch - 3, 1, 1) * 0.5
        m.backbone.body.conv1 = new
        m.transform.image_mean = [0.485, 0.456, 0.406] + [0.5] * (in_ch - 3)
        m.transform.image_std = [0.229, 0.224, 0.225] + [0.25] * (in_ch - 3)
    m.transform.min_size, m.transform.max_size = (CROP,), 1024
    return m.to(DEV)


def sample(npz, s1_dir, stacked, train=True):
    d = np.load(npz)
    rgb, inst = d["rgb"], d["inst"].astype(np.int32)
    x = rgb.astype(np.float32) / 255.0
    if stacked:
        x = np.concatenate([x, np.load(s1_dir / (Path(npz).stem + ".npy")).astype(np.float32) / 255.0], -1)
    if train:
        h, w = inst.shape
        y0, x0 = np.random.randint(0, h - CROP + 1), np.random.randint(0, w - CROP + 1)
        if np.random.rand() < 0.5:
            # crowded-row sampling: centre the crop on a spot where two houses touch
            big = 1 << 30
            mx = ndi.maximum_filter(inst, 5); mn = ndi.minimum_filter(np.where(inst > 0, inst, big), 5)
            ys, xs = np.nonzero((inst > 0) & (mn < big) & (mx != mn))
            if len(ys):
                k = np.random.randint(len(ys))
                y0 = int(np.clip(ys[k] - CROP // 2, 0, h - CROP)); x0 = int(np.clip(xs[k] - CROP // 2, 0, w - CROP))
        x, inst = x[y0:y0 + CROP, x0:x0 + CROP], inst[y0:y0 + CROP, x0:x0 + CROP]
        k = np.random.randint(4)
        x, inst = np.rot90(x, k).copy(), np.rot90(inst, k).copy()
        if np.random.rand() < 0.5:
            x, inst = x[:, ::-1].copy(), inst[:, ::-1].copy()
    boxes, masks = [], []
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        mk = inst == v
        if mk.sum() < 30:
            continue
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]); masks.append(mk)
    tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
           "labels": torch.ones(len(boxes), dtype=torch.int64),
           "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8) if masks else torch.zeros((0, *inst.shape), dtype=torch.uint8)}
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), tgt, inst


def train_maskrcnn(files, s1_dir, stacked, steps=S2_STEPS):
    in_ch = 6 if stacked else 3
    m = build_maskrcnn(in_ch); m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    per_epoch = 100
    total = steps
    # OneCycleLR divides by zero on tiny runs (the 2-step preflight), so only use it on real ones
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=total, pct_start=0.1) if total >= 20 else None
    scaler = torch.amp.GradScaler()
    for step in range(total):
        batch = [sample(random.choice(files), s1_dir, stacked) for _ in range(4)]
        imgs = [b[0].to(DEV) for b in batch]
        tgts = [{k: v.to(DEV) for k, v in b[1].items()} for b in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(imgs, tgts).values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched:
            sched.step()
        if step % per_epoch == 0:
            log(f"  mask r-cnn ({in_ch} ch) step {step}/{total} loss {loss.item():.3f}")
    m.eval()
    return m


@torch.no_grad()
def maskrcnn_labels(m, npz, s1_dir, stacked, thr=0.5):
    x, _, gt = sample(npz, s1_dir, stacked, train=False)
    m.transform.min_size, m.transform.max_size = (x.shape[1],), x.shape[1]
    out = m([x.to(DEV)])[0]
    m.transform.min_size, m.transform.max_size = (CROP,), 1024
    lab = np.zeros(gt.shape, np.int32)
    keep = out["scores"] >= thr
    for i, mk in enumerate((out["masks"][keep][:, 0] > 0.5).cpu().numpy(), start=1):
        lab[mk & (lab == 0)] = i
    return lab, gt


# ------------------------------------------------------------------ main
def main():
    whu = glob.glob("/kaggle/input/**/unet_whu_instance.pt", recursive=True)
    train_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/train/*.npz", recursive=True))
    test_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/test/*.npz", recursive=True))
    log(f"train {len(train_files)} | test {len(test_files)} | whu model: {whu}")
    if not whu or len(train_files) < 10 or len(test_files) < 5:
        raise SystemExit("missing inputs")
    # ---- preflight: every stage on 2 tiles, so a bug costs a minute, not an hour
    pre = TMP / "pre_s1"; pre.mkdir(parents=True, exist_ok=True)
    w = load_unet(whu[0])
    for f in train_files[:2]:
        ps, pe, pi, pd = unet_outputs([w], np.load(f)["rgb"])
        np.save(pre / (f.stem + ".npy"), maps_u8(ps, pe, pi))
        tr.instances_from_prediction(ps, pe, pi, pd)
    for stacked in (False, True):
        mm = train_maskrcnn(train_files[:2], pre, stacked, steps=2)
        lab, gt = maskrcnn_labels(mm, train_files[0], pre, stacked)
        score_tile(lab, gt)
        del mm
    torch.cuda.empty_cache()
    log("preflight OK")

    # ---- stage 1: two folds, out-of-fold maps for every training tile
    xs = np.array([int(f.stem.split("_")[2]) for f in train_files])
    mid = np.median(xs)
    folds = [[f for f, x in zip(train_files, xs) if x < mid], [f for f, x in zip(train_files, xs) if x >= mid]]
    log(f"stage-1 folds by geography: {len(folds[0])} west, {len(folds[1])} east")
    s1_models = []
    for k in (0, 1):
        d = TMP / f"fold{k}"; d.mkdir(exist_ok=True)
        for f in folds[k]:
            (d / f.name).symlink_to(f)
        out = WORK / f"s1_fold{k}.pt"
        subprocess.run([sys.executable, str(WORK / "train.py"), "--tiles", str(d), "--val-tiles", str(test_files[0].parent),
                        "--val-max", "9", "--init", whu[0], "--epochs", str(S1_EPOCHS), "--batch", "8",
                        "--crops", "8", "--seam-boost", "4", "--gsd", "0.30", "--out", str(out)], check=True)
        s1_models.append(out)
    s1 = TMP / "s1"; s1.mkdir(exist_ok=True)
    nets = [load_unet(p) for p in s1_models]
    for k, fold in enumerate(folds):
        other = nets[1 - k]                      # the model that never saw this half
        for f in fold:
            ps, pe, pi, _ = unet_outputs([other], np.load(f)["rgb"])
            np.save(s1 / (f.stem + ".npy"), maps_u8(ps, pe, pi))
    rows_a, panels = [], {}
    for f in test_files:
        d = np.load(f)
        ps, pe, pi, pd = unet_outputs(nets, d["rgb"])
        np.save(s1 / (f.stem + ".npy"), maps_u8(ps, pe, pi))
        lab, _ = tr.instances_from_prediction(ps, pe, pi, pd)
        rows_a.append(score_tile(lab, d["inst"].astype(np.int32)))
        panels.setdefault(f.stem, {})["A"] = lab
    del nets; torch.cuda.empty_cache()
    results = {"A  U-Net + roof edges": summarise(rows_a)}
    log("A:", results["A  U-Net + roof edges"])

    # ---- stage 2: Mask R-CNN alone vs stacked on stage-1 maps
    for name, stacked in (("B  Mask R-CNN, image only", False), ("D  Mask R-CNN, image + U-Net maps (stacked)", True)):
        m = train_maskrcnn(train_files, s1, stacked)
        rows = []
        for f in test_files:
            lab, gt = maskrcnn_labels(m, f, s1, stacked)
            rows.append(score_tile(lab, gt))
            panels[f.stem][name[0]] = lab
        results[name] = summarise(rows)
        log(name, results[name])
        torch.save(m.state_dict(), WORK / f"maskrcnn_{'stacked' if stacked else 'rgb'}.pt")
        del m; torch.cuda.empty_cache()

    (WORK / "fair_results.json").write_text(json.dumps(results, indent=2))
    log("RESULTS\n" + json.dumps(results, indent=2))

    # ---- picture: 3 test tiles, image | label | A | B | D
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = test_files[:3]
    fig, ax = plt.subplots(len(pick), 5, figsize=(26, 5.4 * len(pick)))
    for r, f in enumerate(pick):
        d = np.load(f); rgb = d["rgb"]; gt = d["inst"].astype(np.int32)
        for c, (lab, title) in enumerate([(None, f.stem[:30]), (gt, "dataset label"),
                                          (panels[f.stem]["A"], "A  U-Net + edges"),
                                          (panels[f.stem]["B"], "B  Mask R-CNN"),
                                          (panels[f.stem]["D"], "D  stacked")]):
            ov = rgb.copy()
            if lab is not None:
                rng = np.random.default_rng(3)
                cols = (rng.random((int(lab.max()) + 2, 3)) * 255).astype(np.uint8); cols[0] = 0
                mk = lab > 0
                ov[mk] = (0.5 * ov[mk] + 0.5 * cols[lab[mk]]).astype(np.uint8)
                n = len(np.unique(lab)) - 1
                title = f"{title}: {n} shapes"
            ax[r, c].imshow(ov); ax[r, c].set_title(title, fontsize=11); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.tight_layout(); fig.savefig(WORK / "fair_samples.jpg", dpi=60)
    log("done")


if __name__ == "__main__":
    main()
