"""Bake-off: the teammate's building models vs roofs v1, and two ways of combining them.

    F   teammate's Inria U-Net (binary roof mask), split into houses by watershed
    R   roofs v1: our U-Net -> Mask R-CNN stack (frozen)
    FU  fusion, no training: R's houses, plus roofs only F found
    D+  stacked retrain: Mask R-CNN on RGB + our 3 U-Net maps + F's roof map + the
        teammate's UAVid building map (8 channels), trained on the Gandhinagar
        learning side exactly like roofs v1 was

All four are scored on the same fair Gandhinagar exam (houses never trained on), then
drawn on the teammate's test image and four Indian city layouts.

Also checks whether the teammate's test image is a crop of an Inria training tile
(which would make his result there a memory test, not a generalisation test), and
sanity-checks our loading of his models against Inria ground truth.
"""
import glob, io, json, math, os, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

WORK = Path("/kaggle/working")
ROOFS = Path(glob.glob("/kaggle/input/**/cadastraai-roofs-v1/**/maskrcnn_stacked.pt", recursive=True)[0]).parent
for f in ("compare_fair.py", "train.py"):
    shutil.copy(ROOFS / f, WORK / f)
sys.path.insert(0, str(WORK))
import compare_fair as cf  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

DEV = "cuda"
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
STEPS = int(os.environ.get("S2_STEPS", 1500))
log = cf.log


# ------------------------------------------------------------------ teammate's models
def load_friend(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__
    sd = ck
    if isinstance(ck, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ck and isinstance(ck[k], dict):
                sd = ck[k]; break
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    head = [k for k in sd if k.endswith("segmentation_head.0.weight")]
    n = int(sd[head[0]].shape[0]) if head else 1
    inch = int(sd["encoder.conv1.weight"].shape[1])
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=inch, classes=n)
    miss, unexp = net.load_state_dict(sd, strict=False)
    log(f"teammate model {Path(path).name}: ckpt keys {keys if isinstance(keys, str) else keys[:8]} | "
        f"classes {n} | in_channels {inch} | missing {len(miss)} unexpected {len(unexp)}")
    if len(miss) > 4:
        raise SystemExit(f"architecture mismatch: {miss[:6]}")
    mean = np.array(ck.get("mean", IMNET_MEAN), np.float32)[:3] if isinstance(ck, dict) else IMNET_MEAN
    std = np.array(ck.get("std", IMNET_STD), np.float32)[:3] if isinstance(ck, dict) else IMNET_STD
    return net.to(DEV).eval(), mean, std, n


@torch.no_grad()
def friend_prob(model, rgb, cls=None):
    """probability map for one class (binary models: the single output)"""
    net, mean, std, n = model
    h, w = rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    im = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect").astype(np.float32) / 255
    x = torch.from_numpy(((im - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
    o = net(x).float()[0]
    p = torch.sigmoid(o[0]) if n == 1 else o.softmax(0)[cls if cls is not None else 1]
    return p[:h, :w].cpu().numpy()


def split_houses(mask):
    """binary roof mask -> one id per house (cores as seeds, grown back inside the mask)"""
    seeds, _ = ndi.label(ndi.binary_erosion(mask, iterations=3))
    lab = watershed(-ndi.distance_transform_edt(mask), markers=seeds, mask=mask)
    for v, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is not None and (lab[sl] == v).sum() < 30:
            lab[lab == v] = 0
    return lab.astype(np.int32)


def fuse(r_lab, f_prob, ours_prob):
    """keep every R house; add regions both F and our U-Net call roof that no R house covers"""
    fused = r_lab.copy()
    extra = ((f_prob + ours_prob) / 2 > 0.5) & (r_lab == 0)
    extra = ndi.binary_opening(extra, iterations=2)
    add = split_houses(extra)
    nxt = int(fused.max()) + 1
    for v in np.unique(add):
        if v == 0:
            continue
        m = add == v
        if m.sum() >= 80:
            fused[m] = nxt; nxt += 1
    return fused


# ------------------------------------------------------------------ 8-channel stack
def train_stack(files, s1_dir, in_ch, steps):
    import random
    m = cf.build_maskrcnn(in_ch); m.train()
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=steps, pct_start=0.1) if steps >= 20 else None
    scaler = torch.amp.GradScaler()
    for step in range(steps):
        batch = [cf.sample(random.choice(files), s1_dir, True) for _ in range(4)]
        imgs = [b[0].to(DEV) for b in batch]
        tgts = [{k: v.to(DEV) for k, v in b[1].items()} for b in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(imgs, tgts).values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched:
            sched.step()
        if step % 100 == 0:
            log(f"  D+ ({in_ch} ch) step {step}/{steps} loss {loss.item():.3f}")
    return m.eval()


# ------------------------------------------------------------------ extra test images
Z, N = 19, 4
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
CITIES = [("HSR Layout, Bengaluru", 12.9116, 77.6389), ("Vaishali Nagar, Jaipur", 26.9124, 75.7439),
          ("Dwarka Sector 10, Delhi", 28.5821, 77.0590), ("Chandigarh Sector 22", 30.7333, 76.7794)]


def fetch(lat, lon):
    n = 2 ** Z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    x0, y0 = int(x) - N // 2, int(y) - N // 2
    img = np.zeros((256 * N, 256 * N, 3), np.uint8)
    for j in range(N):
        for i in range(N):
            for _ in range(3):
                try:
                    r = requests.get(URL.format(z=Z, y=y0 + j, x=x0 + i), timeout=20, headers={"User-Agent": "cadastraai-research"})
                    img[j*256:(j+1)*256, i*256:(i+1)*256] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB")); break
                except Exception:
                    time.sleep(1)
    return img


def leak_check(test_rgb):
    """is the teammate's test image a crop of an Inria Austin training tile?"""
    tiles = sorted(glob.glob("/kaggle/input/**/austin*.tif", recursive=True))
    imgs = [t for t in tiles if "/gt/" not in t]
    log(f"leak check: {len(imgs)} Austin image tiles found")
    if not imgs:
        return {"checked": False}
    sift = cv2.SIFT_create(4000)
    q = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2GRAY)
    kq, dq = sift.detectAndCompute(q, None)
    best = (0, None)
    bf = cv2.BFMatcher()
    for t in imgs:
        g = cv2.imread(t, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        kt, dt = sift.detectAndCompute(g, None)
        if dt is None:
            continue
        good = [a for a, b in bf.knnMatch(dq, dt, k=2) if a.distance < 0.7 * b.distance]
        if len(good) < 12:
            continue
        src = np.float32([kq[m.queryIdx].pt for m in good]); dst = np.float32([kt[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        n_in = int(inl.sum()) if inl is not None else 0
        if n_in > best[0]:
            best = (n_in, t)
    in_train = best[1] is not None and os.path.exists(best[1].replace("/images/", "/gt/"))
    res = {"checked": True, "best_match_tile": best[1] and Path(best[1]).name, "inliers": best[0],
           "is_crop_of_inria_tile": best[0] >= 60, "that_tile_has_ground_truth(train split)": in_train}
    log("leak check:", res)
    return res


def sanity_inria(model):
    gts = sorted(glob.glob("/kaggle/input/**/gt/austin*.tif", recursive=True))[:3]
    out = []
    for g in gts:
        im = cv2.cvtColor(cv2.imread(g.replace("/gt/", "/images/")), cv2.COLOR_BGR2RGB)[1000:2024, 1000:2024]
        gt = cv2.imread(g, cv2.IMREAD_GRAYSCALE)[1000:2024, 1000:2024] > 127
        p = friend_prob(model, im) > 0.5
        out.append(round(float((p & gt).sum() / max(1, (p | gt).sum())), 3))
    log("sanity: teammate Inria model on Austin crops (may be his training tiles) IoU:", out)
    return out


def outlines(lab):
    from shapely.geometry import Polygon
    polys = []
    for v in np.unique(lab):
        if v == 0:
            continue
        c, _ = cv2.findContours((lab == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not c:
            continue
        ct = max(c, key=cv2.contourArea)
        if cv2.contourArea(ct) < 30 or len(ct) < 4:
            continue
        p = Polygon(ct.reshape(-1, 2)).buffer(0).simplify(1.0)
        if p.is_empty or p.geom_type != "Polygon":
            continue
        rect = p.minimum_rotated_rectangle
        polys.append(rect if p.area / max(rect.area, 1e-6) > 0.85 else p)
    return polys


def draw(rgb, lab, title):
    vis = rgb.copy()
    polys = outlines(lab)
    for p in polys:
        cv2.polylines(vis, [np.array(p.exterior.coords).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    return vis, f"{title}: {len(polys)}"


# ------------------------------------------------------------------ main
def main():
    tr_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/train/*.npz", recursive=True))
    te_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/test/*.npz", recursive=True))
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    log(f"gandhinagar train {len(tr_files)} test {len(te_files)} | roofs v1 {ROOFS} | teammate {fr}")
    inria = load_friend(fr / "unet_inria_best.pt")
    uavid = load_friend(fr / "uavid_unet_best.pt")
    test_rgb = np.array(Image.open(fr / "test_image_austin.jpg").convert("RGB"))
    report = {}
    for key, fn in (("leak_check", lambda: leak_check(test_rgb)), ("sanity_inria_iou", lambda: sanity_inria(inria))):
        try:
            report[key] = fn()
        except Exception as e:           # optional checks must not sink the bake-off
            report[key] = f"failed: {type(e).__name__}: {e}"
            log(key, "failed:", e)

    nets = [cf.load_unet(ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    r_model = cf.build_maskrcnn(6); r_model.load_state_dict(torch.load(ROOFS / "maskrcnn_stacked.pt", map_location="cpu")); r_model.eval()
    s3, s5 = Path("/kaggle/tmp/s3"), Path("/kaggle/tmp/s5")
    s3.mkdir(parents=True, exist_ok=True); s5.mkdir(parents=True, exist_ok=True)

    def maps(rgb, which_nets):
        ps, pe, pi, pd = cf.unet_outputs(which_nets, rgb)
        fprob = friend_prob(inria, rgb)
        uprob = friend_prob(uavid, rgb, cls=1)
        m3 = cf.maps_u8(ps, pe, pi)
        m5 = np.dstack([m3, (fprob * 255).astype(np.uint8), (uprob * 255).astype(np.uint8)])
        return m3, m5, fprob, ps[1]

    # out-of-fold maps for training tiles, exactly as roofs v1 was built (west/east folds)
    xs = np.array([int(f.stem.split("_")[2]) for f in tr_files]); mid = np.median(xs)
    for f, x in zip(tr_files, xs):
        other = nets[1] if x < mid else nets[0]
        _, m5, _, _ = maps(np.load(f)["rgb"], [other])
        np.save(s5 / (f.stem + ".npy"), m5)
    test_cache = {}
    for f in te_files:
        m3, m5, fprob, oprob = maps(np.load(f)["rgb"], nets)
        np.save(s3 / (f.stem + ".npy"), m3); np.save(s5 / (f.stem + ".npy"), m5)
        test_cache[f.stem] = (fprob, oprob)
    log("maps ready")

    # preflight the 8-channel stack for 2 steps
    train_stack(tr_files[:2], s5, 8, 2)
    log("preflight OK")
    dplus = train_stack(tr_files, s5, 8, STEPS)
    torch.save(dplus.state_dict(), WORK / "maskrcnn_stacked8_dplus.pt")

    rows = {k: [] for k in ("F  teammate Inria U-Net", "R  roofs v1 (ours)", "FU fusion (no training)", "D+ stacked 8-channel")}
    panels = {}
    for f in te_files:
        gt = np.load(f)["inst"].astype(np.int32); rgb = np.load(f)["rgb"]
        fprob, oprob = test_cache[f.stem]
        F_lab = split_houses(fprob > 0.5)
        R_lab, _ = cf.maskrcnn_labels(r_model, f, s3, True)
        FU_lab = fuse(R_lab, fprob, oprob)
        D_lab, _ = cf.maskrcnn_labels(dplus, f, s5, True)
        for k, lab in zip(rows, (F_lab, R_lab, FU_lab, D_lab)):
            rows[k].append(cf.score_tile(lab, gt))
        panels[f.stem] = (rgb, F_lab, R_lab, FU_lab, D_lab)
    report["gandhinagar_exam"] = {k: cf.summarise(v) for k, v in rows.items()}
    log("EXAM\n" + json.dumps(report["gandhinagar_exam"], indent=2))

    # pictures: teammate's image, 4 Indian cities, 1 exam tile
    extra = [("Teammate's test image (Austin?)", test_rgb)] + [(n, fetch(la, lo)) for n, la, lo in CITIES]
    tmp = Path("/kaggle/tmp/vis"); tmp.mkdir(parents=True, exist_ok=True)
    gallery = []
    for name, rgb in extra:
        m3, m5, fprob, oprob = maps(rgb, nets)
        stem = "".join(c for c in name if c.isalnum())[:30]
        np.save(s3 / (stem + ".npy"), m3); np.save(s5 / (stem + ".npy"), m5)
        np.savez(tmp / (stem + ".npz"), rgb=rgb, inst=np.zeros(rgb.shape[:2], np.int16))
        R_lab, _ = cf.maskrcnn_labels(r_model, tmp / (stem + ".npz"), s3, True)
        D_lab, _ = cf.maskrcnn_labels(dplus, tmp / (stem + ".npz"), s5, True)
        gallery.append((name, rgb, split_houses(fprob > 0.5), R_lab, fuse(R_lab, fprob, oprob), D_lab))
    k0 = te_files[len(te_files) // 2].stem
    gallery.append(("Gandhinagar exam tile", *panels[k0]))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    (WORK / "bakeoff").mkdir(exist_ok=True)
    for name, rgb, *labs in gallery:
        fig, ax = plt.subplots(1, 5, figsize=(30, 6.6))
        ax[0].imshow(rgb); ax[0].set_title(name, fontsize=13)
        for c, (lab, t) in enumerate(zip(labs, ("F teammate Inria", "R roofs v1", "FU fusion", "D+ stacked 8-ch")), start=1):
            vis, title = draw(rgb, lab, t)
            ax[c].imshow(vis); ax[c].set_title(title + " roofs", fontsize=13)
        for a in ax: a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(WORK / "bakeoff" / ("".join(ch for ch in name if ch.isalnum())[:40] + ".jpg"), dpi=60)
        plt.close(fig)
    (WORK / "bakeoff_report.json").write_text(json.dumps(report, indent=2))
    log("done")


if __name__ == "__main__":
    main()
