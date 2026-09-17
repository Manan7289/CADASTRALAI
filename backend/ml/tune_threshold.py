"""Pick the detection threshold from data instead of leaving it at 0.5.

0.5 is only the right cut if the training and target domains match, and they do
not: Inria is large, well-separated, mostly light-roofed buildings, while this
AOI is a dense Indian town of small, packed structures under tree cover. In
practice that pushes predicted probabilities down and 0.5 under-detects badly.

So sweep it against the one piece of trustworthy ground truth in the AOI -- the
housing colony whose OSM tagging is actually complete, found the same way
model.py finds it -- and keep the threshold with the best F1.

The probability map is computed once and reused across the sweep, so this costs
one inference pass, not one per threshold.

    python -m ml.tune_threshold
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import model
from ml import infer_buildings as IB
from ml.compare_baseline import colony_bbox, load_polys, score

Image.MAX_IMAGE_PIXELS = None

PROC = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="aoi_image_z18")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--target-gsd", type=float, default=0.3)
    ap.add_argument("--thresholds", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6")
    args = ap.parse_args()

    osm = load_polys(PROC / "buildings.geojson")
    colony = model.find_verified_colony(osm)
    bbox = colony_bbox(colony)
    print(f"ground truth: {len(colony)} buildings in the verified colony\n")

    img, geo = IB.load_geo(args.prefix)
    src_gsd = IB.metres_per_px(geo)
    factor = src_gsd / args.target_gsd
    if abs(factor - 1.0) > 0.05:
        img = np.array(Image.fromarray(img).resize(
            (int(round(geo["width"] * factor)), int(round(geo["height"] * factor))), Image.BICUBIC))
    geo = {**geo, "width": img.shape[1], "height": img.shape[0]}
    m_per_px = IB.metres_per_px(geo)

    mdl, ck, device = IB.load_model()
    print(f"checkpoint val IoU {ck.get('val_iou', float('nan')):.4f} on {device}")
    prob = IB.predict(img, mdl, device, args.batch)
    print("probability map computed; sweeping\n")

    results = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        binary = (prob >= t).astype(np.uint8)
        polys = IB.vectorise(binary, geo, m_per_px)
        s = score(polys, colony, bbox)
        s["threshold"] = t
        s["building_pixel_fraction"] = round(float(binary.mean()), 4)
        results.append(s)
        print(f"  t={t:<5} polys={s['total_predictions']:<5} "
              f"recall={s['building_recall']:<6} prec={s['precision_in_colony']:<6} "
              f"F1={s['f1']:<6} pixels={s['building_pixel_fraction']}")

    best = max(results, key=lambda r: r["f1"])
    print(f"\nbest F1 {best['f1']} at threshold {best['threshold']}")

    (PROC / "threshold_sweep.json").write_text(json.dumps({
        "chosen_threshold": best["threshold"],
        "selected_on": "building-level F1 against the verified OSM colony",
        "checkpoint_val_iou": ck.get("val_iou"),
        "sweep": results,
    }, indent=2), encoding="utf-8")
    print(f"-> threshold_sweep.json (use: python -m ml.infer_buildings --threshold {best['threshold']})")


if __name__ == "__main__":
    main()
