"""Ensemble member 3: Mask R-CNN (torchvision, ResNet50-FPN v2, COCO weights).

On the honest Gandhinagar split this was the strongest single model so far (recall
0.876, outline IoU 0.887). Here it is retrained on RAMP + Gandhinagar at 2x upscale.
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common import (UPSCALE, crops, load_gandhinagar, load_ramp, log, masks_to_labels, pack_instances, panel,
                    score_tile, summarise)

WORK = Path("/kaggle/working")
BUDGET_H = float(os.environ.get("BUDGET_H", 2.6))
DEV = "cuda"
random.seed(0); np.random.seed(0); torch.manual_seed(0)


def build():
    m = maskrcnn_resnet50_fpn_v2(weights="DEFAULT", box_detections_per_img=400,
                                 rpn_pre_nms_top_n_test=4000, rpn_post_nms_top_n_test=2000)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    m.transform.min_size, m.transform.max_size = (256 * UPSCALE,), 2048
    return m.to(DEV)


def sample(s):
    _, rgb, inst, _ = s
    inst = inst.astype(np.int32)
    k = np.random.randint(4)
    rgb, inst = np.rot90(rgb, k).copy(), np.rot90(inst, k).copy()
    if np.random.rand() < 0.5:
        rgb, inst = rgb[:, ::-1].copy(), inst[:, ::-1].copy()
    x = rgb.astype(np.float32) / 255.0
    x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.06, 0.06), 0, 1)
    boxes, masks = [], []
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        mk = inst == v
        if mk.sum() < 12:
            continue
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]); masks.append(mk)
    # the model's transform upsizes the 256 chip to min_size (2x) for both image and targets
    tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
           "labels": torch.ones(len(boxes), dtype=torch.int64),
           "masks": torch.from_numpy(np.stack(masks)).to(torch.uint8) if masks
           else torch.zeros((0, *inst.shape), dtype=torch.uint8)}
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), tgt


@torch.no_grad()
def infer(m, rgb, thr=0.3):
    x = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)).to(DEV)
    m.transform.min_size, m.transform.max_size = (rgb.shape[0] * UPSCALE,), rgb.shape[0] * UPSCALE
    out = m([x])[0]
    keep = out["scores"] >= thr
    ms = list((out["masks"][keep][:, 0] > 0.5).cpu().numpy()); sc = out["scores"][keep].cpu().numpy()
    return ms, sc


def main():
    t0 = time.time()
    tr = load_ramp("train"); random.shuffle(tr)
    gn = crops(load_gandhinagar("train"), 256)
    rep = max(1, int(0.25 * len(tr) / max(1, len(gn))))
    tr = tr[300:]
    pool = tr + gn * rep
    log(f"train pool {len(pool)}")
    m = build(); m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    # steps sized to the time budget (~0.35 s/step at batch 4 on a T4), capped at 3 epochs
    total = int(min(3 * len(pool) / 4, BUDGET_H * 3600 / 0.40))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    log("steps", total)
    for step in range(total):
        batch = [sample(random.choice(pool)) for _ in range(4)]
        imgs = [b[0].to(DEV) for b in batch]
        tgts = [{k: v.to(DEV) for k, v in b[1].items()} for b in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(imgs, tgts).values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 500 == 0:
            log(f"step {step}/{total} loss {loss.item():.3f} ({(time.time() - t0) / 60:.0f} min)")
        if (time.time() - t0) / 3600 > BUDGET_H + 0.3:
            log("time budget reached at step", step); break
    m.eval()
    torch.save(m.state_dict(), WORK / "maskrcnn_ramp_gn.pt")

    results, store = {}, {}
    for name, samples in (("gandhinagar", load_gandhinagar("test")), ("ramp", load_ramp("test"))):
        rows, recs, pics = [], {}, []
        for sid, rgb, inst, partial in samples:
            ms, sc = infer(m, rgb)
            recs[sid] = pack_instances(ms, sc)
            lab = masks_to_labels(ms, sc) if ms else np.zeros(inst.shape, np.int32)
            rows.append(score_tile(lab, inst, partial))
            if len(pics) < 4:
                pics.append((rgb, [None, inst, lab]))
        results[name] = summarise(rows); store[name] = recs
        panel(pics, ["image", "label", "Mask R-CNN"], WORK / f"maskrcnn_{name}.jpg")
        log("maskrcnn", name, results[name])
    np.save(WORK / "pred_maskrcnn.npy", store, allow_pickle=True)
    (WORK / "maskrcnn_results.json").write_text(json.dumps({"test": results}, indent=2))
    log("done in", round((time.time() - t0) / 3600, 2), "h")


if __name__ == "__main__":
    main()
