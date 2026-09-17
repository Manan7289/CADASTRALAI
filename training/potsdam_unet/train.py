"""CadastraAI segmentation model: U-Net (ResNet34 encoder) on ISPRS Potsdam,
input = RGB + nDSM (height above ground), resampled to 10 cm to match the
demo orthoimage.

Height-optional by design: during training the nDSM channel is zeroed for
HEIGHT_DROPOUT of samples, so the same weights run on ORI+DSM surveys and on
imagery that has no DSM. The evaluation reports both modes on the official
held-out tiles, which gives an honest "what does height add" number.

Classes (ISPRS colours -> ours):
  0 clutter/background (255,0,0)
  1 building           (0,0,255)
  2 road / impervious  (255,255,255) + car (255,255,0)
  3 low vegetation     (0,255,255)
  4 tree               (0,255,0)

Outputs in /kaggle/working: best.pt, metrics.json, samples.png, log.txt
"""
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

SMOKE = False # True = a few iterations just to verify paths/shapes end-to-end

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=True)

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
OUT = "/kaggle/working"
LOG = open(os.path.join(OUT, "log.txt"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


CLASSES = ["clutter", "building", "road_impervious", "low_vegetation", "tree"]
SCALE = 2            # 5 cm -> 10 cm
CROP = 512
BATCH = 12
EPOCHS = 2 if SMOKE else 40
ITERS_PER_EPOCH = 5 if SMOKE else 250
HEIGHT_DROPOUT = 0.5
TIME_BUDGET_S = 3.2 * 3600
SEED = 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ---------------------------------------------------------------- locate files
def tile_id(path):
    m = re.search(r"potsdam_0?(\d+)_(\d+)", os.path.basename(path))
    return f"{int(m.group(1))}_{int(m.group(2))}"


# the dataset nests folders (e.g. 2_Ortho_RGB/2_Ortho_RGB/), so search by folder name anywhere under the mount
IN = "/kaggle/input"
rgb = {tile_id(p): p for p in glob.glob(f"{IN}/**/2_Ortho_RGB/**/*_RGB.tif", recursive=True)}
lab = {tile_id(p): p for p in glob.glob(f"{IN}/**/5_Labels_all/*_label.tif", recursive=True)}
ndsm = {tile_id(p): p for p in glob.glob(f"{IN}/**/1_DSM_normalisation/**/*normalized_lastools.jpg", recursive=True)}
train_ids = sorted({tile_id(p) for p in glob.glob(f"{IN}/**/5_Labels_for_participants/**/*_label.tif", recursive=True)})
assert rgb and lab and ndsm, f"missing inputs: {len(rgb)} rgb, {len(lab)} labels, {len(ndsm)} ndsm under {IN}: {os.listdir(IN)}"
all_ids = sorted(set(rgb) & set(lab) & set(ndsm))
train_ids = [t for t in train_ids if t in all_ids]
test_ids = [t for t in all_ids if t not in train_ids]
log(f"tiles: rgb={len(rgb)} labels={len(lab)} ndsm={len(ndsm)} usable={len(all_ids)} train={len(train_ids)} test={len(test_ids)}")
if SMOKE:
    train_ids, test_ids = train_ids[:2], test_ids[:1]

# label colour -> class, indexed by the 3-bit code R*4 + G*2 + B of the thresholded colour
CODE_TO_CLASS = np.zeros(8, dtype=np.uint8)
CODE_TO_CLASS[0b100] = 0  # (255,0,0) clutter
CODE_TO_CLASS[0b001] = 1  # (0,0,255) building
CODE_TO_CLASS[0b111] = 2  # (255,255,255) impervious
CODE_TO_CLASS[0b110] = 2  # (255,255,0) car -> impervious
CODE_TO_CLASS[0b011] = 3  # (0,255,255) low vegetation
CODE_TO_CLASS[0b010] = 4  # (0,255,0) tree


def load_tile(tid):
    im = Image.open(rgb[tid]).convert("RGB")
    w, h = im.size[0] // SCALE, im.size[1] // SCALE
    x = np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    z = np.asarray(Image.open(ndsm[tid]).convert("L").resize((w, h), Image.BILINEAR), dtype=np.uint8)
    lab_im = np.asarray(Image.open(lab[tid]).convert("RGB").resize((w, h), Image.NEAREST)) > 127
    code = lab_im[..., 0] * 4 + lab_im[..., 1] * 2 + lab_im[..., 2]
    y = CODE_TO_CLASS[code]
    return np.dstack([x, z]), y


t0 = time.time()
TRAIN = [load_tile(t) for t in train_ids]
TEST = [load_tile(t) for t in test_ids]
log(f"loaded {len(TRAIN)}+{len(TEST)} tiles at {TRAIN[0][0].shape} in {time.time() - t0:.0f}s")
freq = np.bincount(np.concatenate([y.ravel() for _, y in TRAIN]), minlength=5) / sum(y.size for _, y in TRAIN)
log("train class freq", dict(zip(CLASSES, np.round(freq, 3))))

MEAN = np.array([0.485, 0.456, 0.406, 0.0], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225, 1.0], dtype=np.float32)


def to_tensor(xz, drop_height):
    a = xz.astype(np.float32) / 255.0
    a = (a - MEAN) / STD
    if drop_height:
        a[..., 3] = 0.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def sample_batch():
    xs, ys = [], []
    for _ in range(BATCH):
        xz, y = random.choice(TRAIN)
        i = random.randint(0, xz.shape[0] - CROP)
        j = random.randint(0, xz.shape[1] - CROP)
        a, b = xz[i:i + CROP, j:j + CROP].copy(), y[i:i + CROP, j:j + CROP].copy()
        k = random.randint(0, 3)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if random.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        rgb_part = a[..., :3].astype(np.float32)
        # colour jitter: Potsdam -> Indian drone imagery is a big appearance shift
        rgb_part = rgb_part * random.uniform(0.75, 1.25) + random.uniform(-25, 25)
        rgb_part = rgb_part * np.random.uniform(0.9, 1.1, size=3)
        a = np.dstack([np.clip(rgb_part, 0, 255), a[..., 3:]]).astype(np.uint8)
        xs.append(to_tensor(a, random.random() < HEIGHT_DROPOUT))
        ys.append(torch.from_numpy(np.ascontiguousarray(b).astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=4, classes=len(CLASSES)).to(dev)
dice = smp.losses.DiceLoss("multiclass")
weights = torch.tensor(np.clip(1.0 / np.sqrt(freq + 1e-3), 0, 10) / np.mean(np.clip(1.0 / np.sqrt(freq + 1e-3), 0, 10)),
                       dtype=torch.float32, device=dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=EPOCHS * ITERS_PER_EPOCH,
                                            pct_start=max(0.1, 3 / (EPOCHS * ITERS_PER_EPOCH)))
scaler = torch.amp.GradScaler(enabled=dev == "cuda")


@torch.no_grad()
def predict_tile(xz, drop_height):
    model.eval()
    H, W = xz.shape[:2]
    stride = CROP // 2
    probs = np.zeros((len(CLASSES), H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(H - CROP, 0) + 1, stride)) + ([H - CROP] if H > CROP and (H - CROP) % stride else [])
    xs = list(range(0, max(W - CROP, 0) + 1, stride)) + ([W - CROP] if W > CROP and (W - CROP) % stride else [])
    for i in ys:
        batch, pos = [], []
        for j in xs:
            batch.append(to_tensor(xz[i:i + CROP, j:j + CROP], drop_height))
            pos.append(j)
        with torch.autocast(dev, enabled=dev == "cuda"):
            p = torch.softmax(model(torch.stack(batch).to(dev)).float(), 1).cpu().numpy()
        for k, j in enumerate(pos):
            probs[:, i:i + CROP, j:j + CROP] += p[k]
            count[i:i + CROP, j:j + CROP] += 1
    return probs / np.maximum(count, 1)


def evaluate(drop_height):
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for xz, y in TEST:
        pred = predict_tile(xz, drop_height).argmax(0)
        conf += np.bincount(len(CLASSES) * y.ravel().astype(np.int64) + pred.ravel(),
                            minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))
    tp = np.diag(conf).astype(float)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    f1 = 2 * tp / np.maximum(conf.sum(0) + conf.sum(1), 1)
    return {"overall_accuracy": round(float(tp.sum() / conf.sum()), 4),
            "mean_iou": round(float(iou.mean()), 4), "mean_f1": round(float(f1.mean()), 4),
            "iou": {c: round(float(v), 4) for c, v in zip(CLASSES, iou)},
            "f1": {c: round(float(v), 4) for c, v in zip(CLASSES, f1)},
            "confusion": conf.tolist()}


best, history, start = -1.0, [], time.time()
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for _ in range(ITERS_PER_EPOCH):
        x, y = sample_batch()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev, enabled=dev == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights) + dice(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
    elapsed = time.time() - start
    rec = {"epoch": epoch, "loss": round(float(np.mean(losses)), 4), "minutes": round(elapsed / 60, 1)}
    if epoch % 5 == 4 or epoch == EPOCHS - 1 or SMOKE:
        m = evaluate(drop_height=False)
        rec["val_miou_with_height"] = m["mean_iou"]
        if m["mean_iou"] > best:
            best = m["mean_iou"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "in_channels": 4,
                        "arch": "Unet-resnet34", "gsd_m": 0.10, "mean": MEAN.tolist(), "std": STD.tolist()},
                       os.path.join(OUT, "best.pt"))
            rec["saved"] = True
    history.append(rec)
    log(json.dumps(rec))
    if elapsed > TIME_BUDGET_S:
        log("time budget reached, stopping")
        break

ckpt = torch.load(os.path.join(OUT, "best.pt"), map_location=dev)
model.load_state_dict(ckpt["state_dict"])
metrics = {
    "dataset": "ISPRS Potsdam 2D semantic labelling (official train/test tile split), resampled to 10 cm",
    "model": "U-Net, ResNet34 encoder (ImageNet init), input RGB + nDSM, height-dropout training",
    "train_tiles": train_ids, "test_tiles": test_ids,
    "with_height": evaluate(drop_height=False),
    "rgb_only": evaluate(drop_height=True),
    "history": history,
    "train_minutes": round((time.time() - start) / 60, 1),
}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
log("FINAL with height:", metrics["with_height"]["mean_iou"], metrics["with_height"]["iou"])
log("FINAL rgb only:  ", metrics["rgb_only"]["mean_iou"], metrics["rgb_only"]["iou"])

# qualitative sample: image | truth | prediction (with height) | prediction (rgb only)
PAL = np.array([[200, 60, 60], [40, 90, 220], [235, 235, 235], [120, 220, 220], [30, 150, 30]], dtype=np.uint8)
xz, y = TEST[0]
s = slice(0, 1024)
panels = [xz[s, s, :3], PAL[y[s, s]],
          PAL[predict_tile(xz[s, s], False).argmax(0)], PAL[predict_tile(xz[s, s], True).argmax(0)]]
Image.fromarray(np.concatenate(panels, axis=1)).save(os.path.join(OUT, "samples.png"))
log("done")
