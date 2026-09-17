"""Compare AI-extracted parcels with reference data, in the survey's UTM zone.

Parcel layer comparison (existing GIS record vs AI):
  each AI parcel gets a status --
    MATCH            best reference IoU >= MATCH_IOU (boundary agrees with the record)
    BOUNDARY_DIFFERS best IoU in [PARTIAL_IOU, MATCH_IOU) -- same plot, boundary moved
    SPLIT_OR_MERGED  overlaps reference parcels substantially but no single good match
    NOT_IN_RECORD    no meaningful overlap with any reference parcel (new / unrecorded)
  and each reference parcel with no AI counterpart is MISSING_FROM_AI.

GNSS / ground-truth points (surveyed boundary corners):
  horizontal error = distance from each point to the nearest AI parcel
  boundary vertex; reported as mean, RMSE, CE90 and max, with points beyond
  the tolerance listed. Points typed as something other than a corner
  (e.g. 'control', 'check_inside') are reported but not scored as corners.
"""
import json
import time

import numpy as np
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import nearest_points, transform as shp_transform
from shapely.strtree import STRtree

import reference
import survey

MATCH_IOU = 0.75
PARTIAL_IOU = 0.4
MIN_OVERLAP_FRAC = 0.2
CORNER_TYPES = {"corner", "boundary", "bc", "parcel_corner", "boundary_corner", "pillar", "mark"}


def _load(path, fwd):
    fc = json.loads(path.read_text())
    return [(f["properties"], shp_transform(fwd.transform, shape(f["geometry"]))) for f in fc["features"]]


def compare(sid, tolerance_m=1.0):
    meta, fwd, back = survey._utm(sid)
    d = survey.survey_dir(sid)
    rd = reference.ref_dir(sid)
    ai = _load(d / "parcels.geojson", fwd)
    result = {"compared_at": time.strftime("%Y-%m-%d %H:%M"), "crs": meta["crs"], "tolerance_m": tolerance_m}

    if (rd / "parcels.geojson").exists():
        ref = _load(rd / "parcels.geojson", fwd)
        tree = STRtree([g for _, g in ref])
        overlaps = []                      # per AI parcel: [(ref index, intersection area)]
        for props, g in ai:
            ov = []
            for j in tree.query(g, predicate="intersects"):
                inter = g.intersection(ref[int(j)][1]).area
                if inter > 0:
                    ov.append((int(j), inter))
            overlaps.append(ov)

        # a reference parcel substantially covered by 2+ AI parcels = the AI split it;
        # an AI parcel substantially covering 2+ reference parcels = the AI merged them
        claims = {}
        for i, ov in enumerate(overlaps):
            for j, inter in ov:
                if inter / ref[j][1].area >= MIN_OVERLAP_FRAC:
                    claims.setdefault(j, []).append(i)
        split_ai = {i for j, ais in claims.items() if len(ais) >= 2 for i in ais}

        matched_ref = set()
        per_ai, ious, area_diffs = [], [], []
        counts = {"MATCH": 0, "BOUNDARY_DIFFERS": 0, "SPLIT_OR_MERGED": 0, "NOT_IN_RECORD": 0}
        for i, ((props, g), ov) in enumerate(zip(ai, overlaps)):
            best_iou, best_j = 0.0, None
            for j, inter in ov:
                iou = inter / g.union(ref[j][1]).area
                if iou > best_iou:
                    best_iou, best_j = iou, j
            covered = sum(inter for _, inter in ov)
            frac = covered / g.area if g.area else 0
            merged_refs = sum(1 for j, inter in ov if inter / ref[j][1].area >= MIN_OVERLAP_FRAC and inter / g.area >= MIN_OVERLAP_FRAC)
            if best_iou >= MATCH_IOU:
                st = "MATCH"
            elif merged_refs >= 2 or (i in split_ai and frac >= MIN_OVERLAP_FRAC):
                st = "SPLIT_OR_MERGED"
            elif best_iou >= PARTIAL_IOU:
                st = "BOUNDARY_DIFFERS"
            elif frac >= MIN_OVERLAP_FRAC:
                st = "SPLIT_OR_MERGED"
            else:
                st = "NOT_IN_RECORD"
            counts[st] += 1
            entry = {"parcel_id": props["id"], "status": st, "iou": round(best_iou, 3)}
            if best_j is not None and st != "NOT_IN_RECORD":
                matched_ref.add(best_j)
                rg = ref[best_j][1]
                entry.update({"ref_id": ref[best_j][0]["ref_id"],
                              "area_diff_pct": round(100 * (g.area - rg.area) / rg.area, 1) if rg.area else None,
                              "boundary_offset_m": round(g.hausdorff_distance(rg), 2)})
                if st in ("MATCH", "BOUNDARY_DIFFERS"):
                    ious.append(best_iou)
                    if rg.area:
                        area_diffs.append(abs(g.area - rg.area) / rg.area)
            per_ai.append(entry)

        missing = [{"ref_id": ref[j][0]["ref_id"], "area_m2": round(ref[j][1].area, 1),
                    "geometry": mapping(shp_transform(back.transform, ref[j][1]))}
                   for j in range(len(ref)) if j not in matched_ref
                   and not any(ref[j][1].intersects(g) and ref[j][1].intersection(g).area / ref[j][1].area >= MIN_OVERLAP_FRAC
                               for _, g in ai)]
        n = len(ai) or 1
        result["parcels"] = {
            "ai_parcels": len(ai), "reference_parcels": len(ref), "counts": counts,
            "missing_from_ai": len(missing),
            "agreement_pct": round(100 * counts["MATCH"] / n, 1),
            "mean_iou_matched": round(float(np.mean(ious)), 3) if ious else None,
            "median_area_diff_pct": round(100 * float(np.median(area_diffs)), 1) if area_diffs else None,
            "per_parcel": per_ai, "missing": missing,
            "thresholds": {"match_iou": MATCH_IOU, "partial_iou": PARTIAL_IOU},
        }

    if (rd / "gnss.geojson").exists():
        pts = _load(rd / "gnss.geojson", fwd)
        vertices = []
        for _, g in ai:
            vertices.extend(g.exterior.coords[:-1])
        vtree = MultiPoint(vertices) if vertices else None
        boundaries = STRtree([g.exterior for _, g in ai]) if ai else None
        rows, errs = [], []
        for props, p in pts:
            is_corner = props.get("type", "corner") in CORNER_TYPES
            if vtree is None:
                continue
            nv = nearest_points(p, vtree)[1]
            d_vertex = p.distance(nv)
            j = boundaries.nearest(p)
            nb = nearest_points(p, ai[int(j)][1].exterior)[1]
            d_line = p.distance(nb)
            row = {"point_id": props.get("point_id"), "type": props.get("type"), "is_corner": is_corner,
                   "accuracy_m": props.get("accuracy_m"),
                   "error_to_vertex_m": round(d_vertex, 3), "error_to_boundary_m": round(d_line, 3),
                   "within_tolerance": bool(d_vertex <= tolerance_m),
                   "nearest_vertex": mapping(shp_transform(back.transform, nv)),
                   "point": mapping(shp_transform(back.transform, p))}
            rows.append(row)
            if is_corner:
                errs.append(d_vertex)
        e = np.array(errs)
        result["gnss"] = {
            "points": len(rows), "corner_points": len(errs),
            "mean_error_m": round(float(e.mean()), 3) if e.size else None,
            "rmse_m": round(float(np.sqrt((e ** 2).mean())), 3) if e.size else None,
            "ce90_m": round(float(np.percentile(e, 90)), 3) if e.size else None,
            "max_error_m": round(float(e.max()), 3) if e.size else None,
            "within_tolerance_pct": round(100 * float((e <= tolerance_m).mean()), 1) if e.size else None,
            "rows": rows,
        }

    (reference.ref_dir(sid) / "comparison.json").write_text(json.dumps(result))
    return result


def annotate_parcels(sid, result):
    """Copy each parcel's record-comparison status into its properties so the
    workbench can colour by it and the export carries it."""
    per = {r["parcel_id"]: r for r in (result.get("parcels") or {}).get("per_parcel", [])}
    fc = json.loads((survey.survey_dir(sid) / "parcels.geojson").read_text())
    for f in fc["features"]:
        r = per.get(f["properties"]["id"])
        f["properties"]["record_status"] = r["status"] if r else None
        f["properties"]["record_iou"] = r["iou"] if r else None
        f["properties"]["record_ref_id"] = r.get("ref_id") if r else None
    (survey.survey_dir(sid) / "parcels.geojson").write_text(json.dumps(fc))
