"""Score a checkpoint on the actual target AOI, not on Inria's validation set.

Inria val IoU says how well the model does on more Inria. It says nothing about
whether the model transfers to Igatpuri, and the two came apart badly: the first
model here scored 0.733 on Inria val while finding only 28% of Igatpuri's OSM
buildings.

Two numbers matter and both are reported:

  recall          fraction of OSM-tagged buildings the model actually hits.
  redness gap     mean (R - mean(G,B)) of the roofs it found, minus the same for
                  the roofs it missed. Only near-zero is healthy, and the sign
                  does not matter -- what it measures is whether roof colour
                  predicts detection at all. Positive means the network is keyed
                  on terracotta (the shortcut Inria's Vienna/Tyrol tiles teach);
                  negative means it is keyed on pale concrete instead. Both are
                  the same failure and both break on a roof palette it has not
                  seen.

    python -m ml.eval_aoi --checkpoint ../data/models/inria_unet.pt --threshold 0.3
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import shape
from shapely.strtree import STRtree

from ml import infer_buildings as IB

Image.MAX_IMAGE_PIXELS = None

PROC = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_polys(path: Path):
    fc = json.loads(path.read_text(encoding="utf-8"))
    return [shape(f["geometry"]) for f in fc["features"] if f["geometry"]["type"] == "Polygon"]


def evaluate(pred, osm, img, geo):
    H, W = img.shape[:2]
    lon0, lat0, lon1, lat1 = geo["lon_nw"], geo["lat_nw"], geo["lon_se"], geo["lat_se"]
    tree = STRtree(pred) if pred else None

    found, missed = [], []
    for b in osm:
        hit = bool(tree and any(pred[i].intersects(b) for i in tree.query(b)))
        px = int((b.centroid.x - lon0) / (lon1 - lon0) * W)
        py = int((b.centroid.y - lat0) / (lat1 - lat0) * H)
        x0, x1 = max(0, px - 4), min(W, px + 4)
        y0, y1 = max(0, py - 4), min(H, py + 4)
        if x1 <= x0 or y1 <= y0:
            continue
        (found if hit else missed).append(img[y0:y1, x0:x1].reshape(-1, 3).mean(0))

    def redness(a):
        return float((a[:, 0] - a[:, 1:].mean(1)).mean()) if len(a) else float("nan")

    found, missed = np.array(found), np.array(missed)
    return {
        "osm_buildings": len(osm),
        "found": len(found),
        "missed": len(missed),
        "recall": round(len(found) / max(1, len(osm)), 3),
        "predictions": len(pred),
        "redness_found": round(redness(found), 1),
        "redness_missed": round(redness(missed), 1),
        "redness_gap": round(redness(found) - redness(missed), 1),
        "brightness_found": round(float(found.mean()), 1) if len(found) else None,
        "brightness_missed": round(float(missed.mean()), 1) if len(missed) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prefix", default="aoi_image_z18")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--target-gsd", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    src_img, geo = IB.load_geo(args.prefix)
    src_gsd = IB.metres_per_px(geo)
    factor = src_gsd / args.target_gsd
    work = src_img
    if abs(factor - 1.0) > 0.05:
        work = np.array(Image.fromarray(src_img).resize(
            (int(round(geo["width"] * factor)), int(round(geo["height"] * factor))), Image.BICUBIC))
    wgeo = {**geo, "width": work.shape[1], "height": work.shape[0]}

    model, ck, device = IB.load_model(Path(args.checkpoint) if args.checkpoint else None)
    prob = IB.predict(work, model, device, args.batch)
    pred = IB.vectorise((prob >= args.threshold).astype(np.uint8), wgeo, IB.metres_per_px(wgeo))

    osm = load_polys(PROC / "buildings.geojson")
    out = evaluate(pred, osm, src_img, geo)
    out["checkpoint"] = str(args.checkpoint or "inria_unet.pt")
    out["checkpoint_val_iou"] = ck.get("val_iou")
    out["threshold"] = args.threshold

    print(json.dumps(out, indent=2))
    print(f"\nrecall {out['recall']:.1%} on {out['osm_buildings']} OSM buildings")
    # abs(): a strong negative gap is the same failure as a strong positive one,
    # just keyed on pale roofs instead of terracotta.
    print(f"redness gap {out['redness_gap']:+.1f} "
          f"({'colour-biased' if abs(out['redness_gap']) > 6 else 'healthy'})")
    return out


if __name__ == "__main__":
    main()
