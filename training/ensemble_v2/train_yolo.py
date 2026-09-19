"""Ensemble member 2: YOLOv8-seg, starting from Nacala's trained drone-roof model.

An instance model predicts each house as its own object, so it cannot fuse two
neighbours by construction; it tends to be weaker on exact outlines, which the
U-Net and SAM supply in the fusion step. Trained on the same RAMP + Gandhinagar
data (native 256 px chips written as jpg; imgsz=512 is the 2x upscale).
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from common import (UPSCALE, crops, find, load_gandhinagar, load_ramp, log, pack_instances, panel,  # noqa: E402
                    masks_to_labels, score_tile, summarise)

WORK = Path("/kaggle/working")
DATA = Path("/kaggle/tmp/yolo")
BUDGET_H = float(os.environ.get("BUDGET_H", 2.6))
random.seed(0)


def write_split(samples, split):
    (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
    for name, rgb, inst, _ in samples:
        h, w = inst.shape
        cv2.imwrite(str(DATA / "images" / split / f"{name}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for v in range(1, int(inst.max()) + 1):
            m = (inst == v).astype(np.uint8)
            if m.sum() < 12:
                continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea).reshape(-1, 2)
            if len(c) < 3:
                continue
            lines.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
        (DATA / "labels" / split / f"{name}.txt").write_text("\n".join(lines))


def main():
    t0 = time.time()
    tr = load_ramp("train"); random.shuffle(tr)
    gn = crops(load_gandhinagar("train"), 256)
    rep = max(1, int(0.25 * len(tr) / max(1, len(gn))))
    val, tr = tr[:300], tr[300:]
    write_split(tr + gn * 1, "train")
    # repeats of Gandhinagar crops under new names so the loader samples them more often
    for r in range(1, rep):
        write_split([(f"{n}_r{r}", a, b, c) for n, a, b, c in gn], "train")
    write_split(val, "val")
    (DATA / "data.yaml").write_text(f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: building\n")
    log(f"yolo data written: {len(tr)} RAMP + {len(gn)} x{rep} Gandhinagar")

    w = [p for p in find("nacala/yolo1/**/*.pt")]
    log("nacala yolo weights:", w)
    init = next((p for p in w if "best" in p), w[0] if w else "yolov8m-seg.pt")
    model = YOLO(init)
    log("model:", init, "| task", model.task)
    model.train(data=str(DATA / "data.yaml"), imgsz=256 * UPSCALE, epochs=40, time=BUDGET_H, batch=16,
                workers=4, project=str(WORK / "yolo_runs"), name="nacala_ft", exist_ok=True, patience=100,
                degrees=0, flipud=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=3, overlap_mask=False,
                max_det=400, plots=False, verbose=False)
    best = WORK / "yolo_runs" / "nacala_ft" / "weights" / "best.pt"
    model = YOLO(str(best))

    results, store = {}, {}
    for name, samples in (("gandhinagar", load_gandhinagar("test")), ("ramp", load_ramp("test"))):
        rows, recs, pics = [], {}, []
        for sid, rgb, inst, partial in samples:
            r = model.predict(rgb[..., ::-1].copy(), imgsz=int(max(rgb.shape[:2]) * UPSCALE), conf=0.2, iou=0.6,
                              retina_masks=True, max_det=600, verbose=False)[0]
            if r.masks is not None:
                ms = list(r.masks.data.cpu().numpy() > 0.5); sc = r.boxes.conf.cpu().numpy()
            else:
                ms, sc = [], np.zeros(0)
            recs[sid] = pack_instances(ms, sc)
            lab = masks_to_labels(ms, sc) if ms else np.zeros(inst.shape, np.int32)
            rows.append(score_tile(lab, inst, partial))
            if len(pics) < 4:
                pics.append((rgb, [None, inst, lab]))
        results[name] = summarise(rows); store[name] = recs
        panel(pics, ["image", "label", "YOLOv8-seg"], WORK / f"yolo_{name}.jpg")
        log("yolo", name, results[name])
    np.save(WORK / "pred_yolo.npy", store, allow_pickle=True)
    (WORK / "yolo_results.json").write_text(json.dumps({"init": init, "test": results}, indent=2))
    log("done in", round((time.time() - t0) / 3600, 2), "h")


if __name__ == "__main__":
    main()
