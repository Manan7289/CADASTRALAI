"""Score the RandomForest baseline and the Inria U-Net against the same
ground truth, on the same AOI.

Ground truth is the one housing colony in Igatpuri whose OSM tagging is
actually complete -- found the same way model.py finds it (nearest-neighbour
spacing), so these numbers sit directly beside the ones already in the README.

One asymmetry matters and is not a flaw in the comparison, it is the point:
the RandomForest was *trained on these very buildings*, so its score here is a
best case that it cannot repeat anywhere else. The U-Net has never seen
Igatpuri, or India -- its score is genuine zero-shot transfer from five
European and US cities. A U-Net number merely close to the RandomForest's is
therefore a decisively better result, not a tie.

    python -m ml.compare_baseline
"""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

import model

PROC = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
MARGIN_M = 60.0


def load_polys(path: Path):
    if not path.exists():
        return None
    fc = json.loads(path.read_text(encoding="utf-8"))
    return [shape(f["geometry"]) for f in fc["features"]
            if f["geometry"]["type"] == "Polygon"]


def colony_bbox(polys):
    lons = [c for p in polys for c in p.exterior.coords.xy[0]]
    lats = [c for p in polys for c in p.exterior.coords.xy[1]]
    dlon = MARGIN_M / (111320 * np.cos(np.radians(float(np.mean(lats)))))
    dlat = MARGIN_M / 111320
    return Polygon([(min(lons) - dlon, min(lats) - dlat), (max(lons) + dlon, min(lats) - dlat),
                    (max(lons) + dlon, max(lats) + dlat), (min(lons) - dlon, max(lats) + dlat)])


def score(preds, colony_polys, bbox):
    union = unary_union(colony_polys)
    inside = [p for p in preds if p.centroid.within(bbox)]
    found = sum(1 for b in colony_polys if any(b.intersects(p) for p in inside))
    hits = sum(1 for p in inside if p.intersects(union))
    recall = found / len(colony_polys) if colony_polys else 0.0
    precision = hits / len(inside) if inside else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"total_predictions": len(preds), "predictions_in_colony": len(inside),
            "buildings_found": found, "buildings_total": len(colony_polys),
            "building_recall": round(recall, 3), "precision_in_colony": round(precision, 3),
            "f1": round(f1, 3)}


def main():
    osm = load_polys(PROC / "buildings.geojson")
    colony = model.find_verified_colony(osm)
    bbox = colony_bbox(colony)
    print(f"{len(osm)} OSM buildings in AOI; {len(colony)} in the verified colony\n")

    sources = {
        "RandomForest (model.py, trained ON these buildings)": PROC / "extracted_buildings_rf.geojson",
        "Inria U-Net (zero-shot, never saw India)": PROC / "extracted_buildings_unet.geojson",
    }

    out = {}
    for label, path in sources.items():
        preds = load_polys(path)
        if preds is None:
            print(f"{label}: MISSING ({path.name}) -- skipped")
            continue
        out[label] = score(preds, colony, bbox)
        s = out[label]
        print(f"{label}")
        print(f"   predictions: {s['total_predictions']} total, {s['predictions_in_colony']} in colony")
        print(f"   building recall:      {s['building_recall']}  ({s['buildings_found']}/{s['buildings_total']})")
        print(f"   precision in colony:  {s['precision_in_colony']}")
        print(f"   F1:                   {s['f1']}\n")

    (PROC / "detector_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("-> detector_comparison.json")


if __name__ == "__main__":
    main()
