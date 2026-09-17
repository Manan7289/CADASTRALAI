"""Fine-tune the Potsdam U-Net on Indian drone imagery (Singh Nagar,
Vijayawada) with weak building labels from open footprint datasets.

Starting point: best.pt from the cadastraai-unet-potsdam kernel.

Each step mixes two batches:
  - Potsdam (full 5-class labels, height dropout) -- keeps road / vegetation /
    clutter knowledge and the height pathway from being forgotten;
  - Vijayawada (RGB only, weak labels) -- teaches what a flat, abutting Indian
    concrete roof looks like. Loss is building-vs-rest on labelled pixels only
    (255 = ignore band around footprint edges), with 'not building' pixels
    down-weighted because open footprint layers miss some real buildings.

Evaluation (all on data never trained on):
  - Vijayawada held-out demo block: pixel building IoU/F1 and footprint-level
    recall, before vs after fine-tuning;
  - Potsdam official test tiles: mean IoU, to show nothing was forgotten.

Outputs in /kaggle/working: finetuned.pt, metrics.json, holdout_compare.jpg, log.txt
"""
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

SMOKE = True

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.5.0"], check=True)

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
OUT = "/kaggle/working"
IN = "/kaggle/input"
LOG = open(os.path.join(OUT, "log.txt"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


CLASSES = ["clutter", "building", "road_impervious", "low_vegetation", "tree"]
CROP, BATCH_P, BATCH_V = 512, 6, 6
EPOCHS = 1 if SMOKE else 14
ITERS = 5 if SMOKE else 200
LR = 1e-4
NEG_WEIGHT = 1.0
SEAM_WEIGHT = 3.0          # pixels between touching footprints: teach a gap between neighbours
VEG_PROB = 0.6             # teacher confidence needed for a vegetation pseudo-label
VEG_EXG = 12               # and the pixel must actually look green (2G - R - B)
HEIGHT_DROPOUT = 0.5
POTSDAM_TRAIN_TILES = 2 if SMOKE else 14
POTSDAM_TEST_TILES = 1 if SMOKE else 6
SEED = 11
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
log("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ---------------------------------------------------------------- inputs
ckpt_path = (glob.glob(f"{IN}/**/best.pt", recursive=True) or [None])[0]
assert ckpt_path, f"best.pt not found under {IN}: {os.listdir(IN)}"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
MEAN, STD = np.array(ckpt["mean"], np.float32), np.array(ckpt["std"], np.float32)
log("start checkpoint", ckpt_path)


def tile_id(path):
    m = re.search(r"potsdam_0?(\d+)_(\d+)", os.path.basename(path))
    return f"{int(m.group(1))}_{int(m.group(2))}"


rgb = {tile_id(p): p for p in glob.glob(f"{IN}/**/2_Ortho_RGB/**/*_RGB.tif", recursive=True)}
lab = {tile_id(p): p for p in glob.glob(f"{IN}/**/5_Labels_all/*_label.tif", recursive=True)}
ndsm = {tile_id(p): p for p in glob.glob(f"{IN}/**/1_DSM_normalisation/**/*normalized_lastools.jpg", recursive=True)}
p_train_ids = sorted({tile_id(p) for p in glob.glob(f"{IN}/**/5_Labels_for_participants/**/*_label.tif", recursive=True)})
p_all = sorted(set(rgb) & set(lab) & set(ndsm))
p_train_ids = [t for t in p_train_ids if t in p_all]
p_test_ids = [t for t in p_all if t not in p_train_ids]
random.shuffle(p_train_ids)
p_train_ids, p_test_ids = p_train_ids[:POTSDAM_TRAIN_TILES], p_test_ids[:POTSDAM_TEST_TILES]

CODE_TO_CLASS = np.zeros(8, dtype=np.uint8)
CODE_TO_CLASS[0b100], CODE_TO_CLASS[0b001], CODE_TO_CLASS[0b111] = 0, 1, 2
CODE_TO_CLASS[0b110], CODE_TO_CLASS[0b011], CODE_TO_CLASS[0b010] = 2, 3, 4


def load_potsdam(tid):
    im = Image.open(rgb[tid]).convert("RGB")
    w, h = im.size[0] // 2, im.size[1] // 2
    x = np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.uint8)
    z = np.asarray(Image.open(ndsm[tid]).convert("L").resize((w, h), Image.BILINEAR), dtype=np.uint8)
    l = np.asarray(Image.open(lab[tid]).convert("RGB").resize((w, h), Image.NEAREST)) > 127
    return np.dstack([x, z]), CODE_TO_CLASS[l[..., 0] * 4 + l[..., 1] * 2 + l[..., 2]]


vj_dir = os.path.dirname(os.path.dirname((glob.glob(f"{IN}/**/train/vj_*_label.png", recursive=True) or [""])[0]))
assert vj_dir, "Vijayawada fine-tune dataset not found"
vj_paths = sorted(glob.glob(f"{vj_dir}/train/*_label.png"))[: 8 if SMOKE else None]
random.Random(SEED).shuffle(vj_paths)
n_val = 2 if SMOKE else 12   # validation tiles pick the checkpoint; the demo block stays untouched until the end


def load_vj(p):
    return np.asarray(Image.open(p.replace("_label.png", ".jpg")).convert("RGB")), np.asarray(Image.open(p))


vj_val = [load_vj(p) for p in vj_paths[:n_val]]
vj_train = [load_vj(p) for p in vj_paths[n_val:]]
hold_inst = np.asarray(Image.open(f"{vj_dir}/holdout/demo_block_instances.png")).astype(np.int32)
hold_img = np.asarray(Image.open(f"{vj_dir}/holdout/demo_block.jpg").convert("RGB"))
hold_lab = np.asarray(Image.open(f"{vj_dir}/holdout/demo_block_label.png"))
P_TRAIN = [load_potsdam(t) for t in p_train_ids]
P_TEST = [load_potsdam(t) for t in p_test_ids]
log(f"potsdam train {len(P_TRAIN)} test {len(P_TEST)} | vijayawada train {len(vj_train)} val {len(vj_val)} | holdout {hold_img.shape}")


def to_tensor(xz, drop_height):
    a = (xz.astype(np.float32) / 255.0 - MEAN) / STD
    if drop_height:
        a[..., 3] = 0.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def augment(a, b):
    k = random.randint(0, 3)
    a, b = np.rot90(a, k), np.rot90(b, k)
    if random.random() < 0.5:
        a, b = a[:, ::-1], b[:, ::-1]
    rgbp = a[..., :3].astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    rgbp = rgbp * np.random.uniform(0.92, 1.08, size=3)
    return np.dstack([np.clip(rgbp, 0, 255), a[..., 3:]]).astype(np.uint8), b


def potsdam_batch():
    xs, ys = [], []
    for _ in range(BATCH_P):
        xz, y = random.choice(P_TRAIN)
        i, j = random.randint(0, xz.shape[0] - CROP), random.randint(0, xz.shape[1] - CROP)
        a, b = augment(xz[i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP])
        xs.append(to_tensor(a, random.random() < HEIGHT_DROPOUT))
        ys.append(torch.from_numpy(np.ascontiguousarray(b).astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


def vj_batch():
    xs, ys = [], []
    while len(xs) < BATCH_V:
        img, y = random.choice(vj_train)
        i, j = random.randint(0, img.shape[0] - CROP), random.randint(0, img.shape[1] - CROP)
        yy = y[i:i + CROP, j:j + CROP]
        if (yy != 255).mean() < 0.3:
            continue
        a, b = augment(np.dstack([img[i:i + CROP, j:j + CROP], np.zeros((CROP, CROP), np.uint8)]), yy)
        xs.append(to_tensor(a, True))
        ys.append(torch.from_numpy(np.ascontiguousarray(b).astype(np.int64)))
    return torch.stack(xs), torch.stack(ys)


def build_model():
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=len(CLASSES))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(dev)


@torch.no_grad()
def predict(model, xz, drop_height):
    model.eval()
    H, W = xz.shape[:2]
    stride = CROP // 2
    probs = np.zeros((len(CLASSES), H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ys = sorted(set(list(range(0, max(H - CROP, 0) + 1, stride)) + [max(H - CROP, 0)]))
    xs = sorted(set(list(range(0, max(W - CROP, 0) + 1, stride)) + [max(W - CROP, 0)]))
    for i in ys:
        batch = [to_tensor(xz[i:i + CROP, j:j + CROP], drop_height) for j in xs]
        with torch.autocast(dev, enabled=dev == "cuda"):
            p = torch.softmax(model(torch.stack(batch).to(dev)).float(), 1).cpu().numpy()
        for k, j in enumerate(xs):
            probs[:, i:i + CROP, j:j + CROP] += p[k]
            cnt[i:i + CROP, j:j + CROP] += 1
    return probs / np.maximum(cnt, 1)


def eval_holdout(model, image=None, label=None):
    is_holdout = label is None
    image = hold_img if image is None else image
    label = hold_lab if label is None else label
    xz = np.dstack([image, np.zeros(image.shape[:2], np.uint8)])
    pred_cls = predict(model, xz, True).argmax(0)
    pred_b = pred_cls == 1
    hold_lab_ = label
    m = hold_lab_ != 255
    t, p = hold_lab_[m] == 1, pred_b[m]
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    road = hold_lab_ == 2
    notb = hold_lab_ == 0
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    # footprint-level: a labelled building counts as found if >=50% of its core pixels are predicted building
    comps, n = ndi.label(hold_lab_ == 1)
    found = sum(1 for c in range(1, n + 1) if pred_b[comps == c].mean() >= 0.5)
    return {"pixel_iou_building": round(iou, 4), "pixel_f1_building": round(f1, 4),
            "pixel_precision_vs_open_footprints": round(tp / max(tp + fp, 1), 4),
            "pixel_recall": round(tp / max(tp + fn, 1), 4),
            "footprints_found": found, "footprints_total": n, "footprint_recall": round(found / max(n, 1), 4),
            "predicted_building_fraction_labelled": round(float(p.mean()), 4),
            "label_building_fraction_labelled": round(float(t.mean()), 4),
            "osm_road_pixels_predicted_building": round(float(pred_b[road].mean()), 4) if road.any() else None,
            "seam_pixels_predicted_building": round(float(pred_b[hold_lab_ == 5].mean()), 4) if (hold_lab_ == 5).any() else None,
            **(instance_metrics(pred_b) if is_holdout else {}),
            **({"vegetation_pixels_predicted_vegetation": round(float(np.isin(pred_cls[hold_veg], (3, 4)).mean()), 4)}
               if is_holdout and hold_veg.any() else {}),
            "not_building_pixels_predicted_building": round(float(pred_b[notb].mean()), 4) if notb.any() else None}, pred_b


def instance_metrics(pred_b):
    """How well predicted buildings separate individual footprints: a footprint
    is 'merged' when the predicted component covering most of it also covers
    most of another footprint."""
    comps, ncomp = ndi.label(pred_b)
    ids = np.unique(hold_inst[hold_inst > 0])
    owner = {}
    for i in ids:
        px = comps[hold_inst == i]
        if px.size < 1200:          # < 12 m2 at 10 cm
            continue
        vals = np.bincount(px)
        vals[0] = 0
        c = int(vals.argmax())
        if vals[c] >= 0.5 * px.size:
            owner[int(i)] = c
    by_comp = {}
    for i, c in owner.items():
        by_comp.setdefault(c, []).append(i)
    merged = sum(len(v) for v in by_comp.values() if len(v) > 1)
    return {"footprints_scored": len(owner), "footprints_merged_with_neighbour": merged,
            "footprint_merge_rate": round(merged / max(len(owner), 1), 4), "predicted_components": int(ncomp)}


def eval_val(model):
    """Pooled building F1 over the validation tiles (used only to pick the checkpoint).
    OSM road / not-building pixels count as negatives, so painting lanes as
    building lowers this score."""
    tp = fp = fn = 0
    for img, lab_ in vj_val:
        xz = np.dstack([img, np.zeros(img.shape[:2], np.uint8)])
        pb = predict(model, xz, True).argmax(0) == 1
        m = lab_ != 255
        t, p = lab_[m] == 1, pb[m]
        tp += int((t & p).sum()); fp += int((~t & p).sum()); fn += int((t & ~p).sum())
    return round(2 * tp / max(2 * tp + fp + fn, 1), 4)


def eval_potsdam(model, drop_height):
    conf = np.zeros((5, 5), np.int64)
    for xz, y in P_TEST:
        pr = predict(model, xz, drop_height).argmax(0)
        conf += np.bincount(5 * y.ravel().astype(np.int64) + pr.ravel(), minlength=25).reshape(5, 5)
    tp = np.diag(conf).astype(float)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    return {"mean_iou": round(float(iou.mean()), 4), "iou": {c: round(float(v), 4) for c, v in zip(CLASSES, iou)},
            "overall_accuracy": round(float(tp.sum() / conf.sum()), 4)}


model = build_model()


def add_vegetation_pseudo_labels(teacher, items):
    """Where the weak label says 'not building' (0), let the Potsdam teacher mark
    confident tree (3) / low vegetation (4), but only on pixels that look green."""
    n3 = n4 = 0
    for k, (img, lab_) in enumerate(items):
        pr = predict(teacher, np.dstack([img, np.zeros(img.shape[:2], np.uint8)]), True)
        rgbf = img.astype(np.int16)
        exg = 2 * rgbf[..., 1] - rgbf[..., 0] - rgbf[..., 2]
        lab2 = lab_.copy()
        free = (lab_ == 0) & (exg > VEG_EXG)
        tree = free & (pr[4] > VEG_PROB)
        low = free & ~tree & (pr[3] > VEG_PROB)
        lab2[tree], lab2[low] = 3, 4
        n3 += int(tree.sum()); n4 += int(low.sum())
        items[k] = (img, lab2)
    return n3, n4


t_pl = time.time()
veg_train = add_vegetation_pseudo_labels(model, vj_train)
veg_val = add_vegetation_pseudo_labels(model, vj_val)
hold_veg_ref = [(hold_img, hold_lab.copy())]
add_vegetation_pseudo_labels(model, hold_veg_ref)
hold_veg = np.isin(hold_veg_ref[0][1], (3, 4))
log(f"vegetation pseudo-labels: train tree/low {veg_train}, val {veg_val}, holdout veg px {int(hold_veg.sum())} ({time.time() - t_pl:.0f}s)")
before_hold, before_pred = eval_holdout(model)
log("BEFORE val building F1", eval_val(model))
before_potsdam = eval_potsdam(model, False)
log("BEFORE holdout", json.dumps(before_hold))
log("BEFORE potsdam (with height)", json.dumps(before_potsdam))

dice = smp.losses.DiceLoss("multiclass")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * ITERS)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
best, history, start = -1.0, [], time.time()
for epoch in range(EPOCHS):
    model.train()
    lp, lv = [], []
    for _ in range(ITERS):
        xp, yp = potsdam_batch()
        xv, yv = vj_batch()
        xp, yp, xv, yv = xp.to(dev), yp.to(dev), xv.to(dev), yv.to(dev)
        with torch.autocast(dev, enabled=dev == "cuda"):
            logits = model(torch.cat([xp, xv]))
            lpots = F.cross_entropy(logits[:BATCH_P], yp) + dice(logits[:BATCH_P], yp)
            logp = torch.log_softmax(logits[BATCH_P:].float(), 1)
            log_b = logp[:, 1]
            log_not_b = torch.log1p(-log_b.exp().clamp(max=1 - 1e-6))
            pos, neg, road, tree, low, seam = yv == 1, yv == 0, yv == 2, yv == 3, yv == 4, yv == 5
            # buildings -> building; OSM roads -> road; green teacher labels -> tree / low vegetation;
            # seams between touching footprints and other labelled land -> "not building"
            lvj = -(log_b[pos].sum() + logp[:, 2][road].sum() + logp[:, 4][tree].sum() + logp[:, 3][low].sum()
                    + NEG_WEIGHT * log_not_b[neg].sum() + SEAM_WEIGHT * log_not_b[seam].sum()) / \
                max(1, int(pos.sum() + road.sum() + tree.sum() + low.sum() + NEG_WEIGHT * neg.sum() + SEAM_WEIGHT * seam.sum()))
            loss = lpots + lvj
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        lp.append(lpots.item()); lv.append(lvj.item())
    rec = {"epoch": epoch, "loss_potsdam": round(float(np.mean(lp)), 4), "loss_vijayawada": round(float(np.mean(lv)), 4),
           "minutes": round((time.time() - start) / 60, 1)}
    if epoch % 2 == 1 or epoch == EPOCHS - 1:
        vf1 = eval_val(model)
        rec["val_building_f1"] = vf1
        if vf1 > best:
            best = vf1
            torch.save({**ckpt, "state_dict": model.state_dict(), "finetuned_on": "Vijayawada weak footprints"},
                       os.path.join(OUT, "finetuned.pt"))
            rec["saved"] = True
    history.append(rec)
    log(json.dumps(rec))

model.load_state_dict(torch.load(os.path.join(OUT, "finetuned.pt"), map_location=dev, weights_only=False)["state_dict"])
after_hold, after_pred = eval_holdout(model)
metrics = {
    "start_checkpoint": "cadastraai-unet-potsdam best.pt",
    "holdout": "Singh Nagar demo block (~250 m), excluded from training with a 30 m buffer; labels = open building footprints (weak, incomplete)",
    "holdout_before": before_hold, "holdout_after": after_hold,
    "potsdam_test_before_with_height": before_potsdam,
    "potsdam_test_after_with_height": eval_potsdam(model, False),
    "potsdam_test_after_rgb_only": eval_potsdam(model, True),
    "val_building_f1_best": best,
    "note": "Checkpoint chosen on 12 separate Vijayawada validation tiles; the demo block is scored only before and after. Negatives include OSM roads/lanes and land outside dilated footprints. Open footprints miss some buildings and merge neighbours, so treat pixel metrics as indicative and always check holdout_compare.jpg.",
    "history": history, "train_minutes": round((time.time() - start) / 60, 1),
}
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
log("AFTER holdout", json.dumps(after_hold))

vis = lambda mask: np.where(mask[..., None], (0.45 * hold_img + np.array([0, 110, 255]) * 0.55), hold_img).astype(np.uint8)
row = np.concatenate([hold_img, vis(before_pred), vis(after_pred)], 1)
Image.fromarray(row).resize((row.shape[1] // 3, row.shape[0] // 3)).save(os.path.join(OUT, "holdout_compare.jpg"), quality=88)
log("done")
