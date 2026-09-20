"""Fine-tune the parcel-boundary model on hand-labelled Indian plots.

v1 learned parcel boundaries from Dutch cadastre only. Indian plots look different: thinner
compound walls, party walls between attached houses, trees over boundaries, dust. This fine-tunes
v1 on hand-drawn Indian plot boundaries (training/parcel_boundary/india_labels: Singh Nagar
drone imagery, Jaipur and HSR Layout satellite imagery, all at 0.3 m), mixed 50/50 with Dutch
replay tiles so it keeps what it learned. Loss is masked to the labelled region of each crop.

Scored on held-out Indian crops (boundary F at 1 m, inside the labelled region), before and after.
Inputs: kernel output of cadastraai-parcel-boundary (v1 weights), dataset cadastraai-india-boundary,
dataset cadastraai-demo-rgb (survey images to predict boundary maps for).
"""
import glob, json, os, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/kaggle/working")
import train_boundary as tb  # noqa: E402

WORK = Path("/kaggle/working")
STEPS = int(os.environ.get("STEPS", 2000))
REPLAY_PER_CITY = int(os.environ.get("REPLAY_PER_CITY", 6))
CROP = 256
INDIA_WEIGHT_SHARE = 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def aug_india(rgb, b, reg):
    """Indian crops are small (160-266 px): pad to CROP with the region marked unlabelled."""
    k = random.randint(0, 3)
    rgb, b, reg = np.rot90(rgb, k), np.rot90(b, k), np.rot90(reg, k)
    if random.random() < 0.5:
        rgb, b, reg = rgb[:, ::-1], b[:, ::-1], reg[:, ::-1]
    h, w = b.shape
    ph, pw = max(0, CROP - h), max(0, CROP - w)
    rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect"); b = np.pad(b, ((0, ph), (0, pw))); reg = np.pad(reg, ((0, ph), (0, pw)))
    y, x = random.randint(0, rgb.shape[0] - CROP), random.randint(0, rgb.shape[1] - CROP)
    rgb = rgb[y:y + CROP, x:x + CROP].astype(np.float32); b = b[y:y + CROP, x:x + CROP].astype(np.float32); reg = reg[y:y + CROP, x:x + CROP].astype(np.float32)
    rgb = rgb * np.random.uniform(0.85, 1.15, 3) + np.random.uniform(-15, 15, 3)
    if random.random() < 0.3:
        rgb = cv2.GaussianBlur(rgb, (0, 0), random.uniform(0.3, 1.0))
    return np.clip(rgb, 0, 255), b, reg


def batch(india, dutch, n):
    xs, ys, ms = [], [], []
    for i in range(n):
        if i < n * INDIA_WEIGHT_SHARE or not dutch:
            d = np.load(random.choice(india)); rgb, b, reg = aug_india(d["rgb"], d["b"], d["region"])
        else:
            d = np.load(random.choice(dutch)); rgb, b = tb.augment(d["rgb"], d["b"])
            rgb, b = rgb[:CROP, :CROP], b[:CROP, :CROP]; reg = np.ones_like(b)
        xs.append(((rgb / 255 - tb.MEAN) / tb.STD).transpose(2, 0, 1)); ys.append(b[None]); ms.append(reg[None])
    t = lambda a: torch.from_numpy(np.stack(a)).float().to(tb.DEV)
    return t(xs), t(ys), t(ms)


def score(net, files):
    tol = round(1.0 / tb.GSD)
    from skimage.morphology import skeletonize
    rows = []
    for f in files:
        d = np.load(f)
        prob = tb.predict(net, d["rgb"])
        reg = d["region"] > 0
        pred = skeletonize(prob > 0.5) & reg
        gt = skeletonize(d["b"] > 0) & reg
        dg = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3)
        dp = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
        p = float((dg[pred] <= tol).mean()) if pred.any() else 0.0
        r = float((dp[gt] <= tol).mean()) if gt.any() else 0.0
        rows.append({"crop": Path(f).stem, "precision": round(p, 3), "recall": round(r, 3), "f": round(2 * p * r / max(p + r, 1e-6), 3)})
    m = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("precision", "recall", "f")}
    return m, rows


def main():
    import segmentation_models_pytorch as smp
    v1 = glob.glob("/kaggle/input/**/parcel_boundary_unet.pt", recursive=True)[0]
    india = sorted(glob.glob("/kaggle/input/**/train_*.npz", recursive=True))
    test = sorted(glob.glob("/kaggle/input/**/test_*.npz", recursive=True))
    log(f"v1 {v1} | india train {len(india)} test {len(test)}")
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    net.load_state_dict(torch.load(v1, map_location="cpu", weights_only=False)["state_dict"]); net = net.to(tb.DEV).eval()
    before, rows_b = score(net, test)
    log("BEFORE (v1, Dutch only)", before, rows_b)

    dutch = []
    for i, (c, box) in enumerate(tb.CITIES.items()):
        if c != tb.TEST_CITY:
            dutch += tb.build(c, box, REPLAY_PER_CITY, seed=100 + i)
    log("dutch replay tiles", len(dutch))

    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(STEPS, 1))
    scaler = torch.amp.GradScaler()
    pw = torch.tensor(8.0).to(tb.DEV)
    for step in range(STEPS):
        x, y, m = batch(india, dutch, 8)
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x).float()
        bce = (F.binary_cross_entropy_with_logits(o, y, pos_weight=pw, reduction="none") * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(o) * m
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
        loss = bce + dice
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 250 == 0:
            log(f"step {step}/{STEPS} loss {loss.item():.3f}")
    net.eval()
    after, rows_a = score(net, test)
    log("AFTER (fine-tuned on India)", after, rows_a)
    torch.save({"state_dict": net.state_dict(), "arch": "unet_resnet34", "gsd_m": tb.GSD, "mean": tb.MEAN.tolist(), "std": tb.STD.tolist(),
                "trained_on": "v1 (Dutch cadastre) fine-tuned on hand-labelled Indian plots + Dutch replay"},
               WORK / "parcel_boundary_unet_india.pt")
    (WORK / "india_metrics.json").write_text(json.dumps({"before": before, "after": after, "per_crop_before": rows_b, "per_crop_after": rows_a,
                                                         "steps": STEPS, "train_crops": len(india), "test_crops": len(test)}, indent=2))
    out = WORK / "survey_boundaries"; out.mkdir(exist_ok=True)
    vis = WORK / "vis"; vis.mkdir(exist_ok=True)
    for f in sorted(glob.glob("/kaggle/input/**/cadastraai-demo-rgb/**/*.npz", recursive=True)):
        d = np.load(f, allow_pickle=False); prob = tb.predict(net, d["rgb"])
        np.savez_compressed(out / Path(f).name, boundary=(prob * 255).astype(np.uint8))
        pr = d["rgb"].copy(); pr[prob > 0.5] = (255, 0, 255)
        cv2.imwrite(str(vis / f"survey_{Path(f).stem}.jpg"), cv2.cvtColor(np.hstack([d["rgb"], pr]), cv2.COLOR_RGB2BGR)[::2, ::2])
    for f in test:
        d = np.load(f); prob = tb.predict(net, d["rgb"])
        ov = d["rgb"].copy(); ov[d["b"] > 0] = (255, 255, 0); pr = d["rgb"].copy(); pr[prob > 0.5] = (255, 0, 255)
        cv2.imwrite(str(vis / f"{Path(f).stem}.jpg"), cv2.cvtColor(cv2.resize(np.hstack([ov, pr]), None, fx=2, fy=2), cv2.COLOR_RGB2BGR))
    log("done")


if __name__ == "__main__":
    main()
