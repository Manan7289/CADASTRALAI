"""Turn a surveyor's approved parcels into training labels for the parcel-boundary model.

Every parcel a surveyor approves (as drawn, or after editing) is a verified plot boundary, which
is exactly what the boundary model learns from. This writes the survey image and those boundaries
at the model's 0.3 m scale, with a mask of where labels exist (the approved parcels and a thin
margin), in the same format as training/parcel_boundary/india_labels -- so the model can be
fine-tuned on each finished survey (training/parcel_boundary/finetune_india.py).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import survey

OUT = Path(__file__).resolve().parent.parent / "data" / "training_labels"
GSD_M = 0.3


def export(sid):
    d = survey.survey_dir(sid)
    meta = json.loads((d / "meta.json").read_text())
    fc = json.loads((d / "parcels.geojson").read_text())
    ok = [f for f in fc["features"] if f["properties"].get("status") == "approved"]
    if not ok:
        raise ValueError("No approved parcels yet: approve parcels in Review first.")
    img = Image.open(d / "ori.webp").convert("RGB")
    (s_, w_), (n_, e_) = meta["bounds"]
    # resample the display image so a pixel is about 0.3 m on the ground
    width_m = (e_ - w_) * 111320 * math.cos(math.radians((s_ + n_) / 2))
    sc = (width_m / GSD_M) / img.width
    img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
    W, H = img.size
    my = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    def px(lon, lat):
        return ((lon - w_) / (e_ - w_) * W, (my(n_) - my(lat)) / (my(n_) - my(s_)) * H)
    b = np.zeros((H, W), np.uint8)
    region = np.zeros((H, W), np.uint8)
    for f in ok:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            ring = np.round(np.array([px(*c) for c in poly[0]])).astype(np.int32)
            cv2.fillPoly(region, [ring], 1)
            cv2.polylines(b, [ring], True, 1, 2)
    region = cv2.dilate(region, np.ones((5, 5), np.uint8))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"train_{sid}.npz"
    np.savez_compressed(path, rgb=np.array(img), b=b, region=region)
    return {"file": str(path.relative_to(OUT.parent.parent)), "approved_parcels": len(ok), "size_px": [W, H]}
