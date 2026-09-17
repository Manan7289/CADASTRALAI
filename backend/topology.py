"""Automated topology validation for a parcel layer -- the PS's "detection of
overlapping or inconsistent parcel geometries" and "automated topology
validation module".

All checks run in a projected metre CRS (see export.utm_crs_for), never in
degrees. Each issue carries the parcel id(s), a type, a severity, the
problem geometry (so the dashboard can draw it), and a suggested fix.

Checks:
  INVALID       self-intersecting / malformed ring (GEOS is_valid)
  MULTIPART     one parcel id made of several disjoint pieces
  OVERLAP       two parcels share area (> OVERLAP_TOL_M2)
  GAP           unclaimed hole enclosed by parcels (> GAP_TOL_M2)
  SLIVER        long thin shape (thinness 4*pi*A/P^2 below SLIVER_THINNESS)
  TOO_SMALL     below MIN_PARCEL_M2 -- not a plausible plot
  DUPLICATE     two parcels with (near-)identical geometry

auto_fix() applies the suggested fixes in a deterministic order and returns
the corrected layer plus a log of what changed, so every automatic edit is
reviewable rather than silent.
"""
import math

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

OVERLAP_TOL_M2 = 0.5
GAP_TOL_M2 = 1.0
MAX_GAP_M2 = 400          # bigger enclosed voids are real open land, not a digitising gap
SLIVER_THINNESS = 0.12    # a square is 0.785, a 1:20 strip is ~0.15
SLIVER_MAX_M2 = 60        # large thin parcels (e.g. a lane) are allowed to be thin
MIN_PARCEL_M2 = 12
DUPLICATE_IOU = 0.95


def thinness(geom):
    return 4 * math.pi * geom.area / (geom.length ** 2) if geom.length > 0 else 0.0


def _polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [g for part in geom.geoms for g in _polygons(part)]
    return []


def _issue(kind, severity, ids, geom, msg, fix):
    return {"type": kind, "severity": severity, "parcel_ids": list(ids),
            "geometry": geom, "message": msg, "suggested_fix": fix}


def validate(parcels, exclude=None):
    """parcels: list of {"id": ..., "geometry": shapely Polygon in metres}.
    exclude: optional geometry (e.g. the road/lane corridors) that is allowed
    to sit between parcels, so enclosed roads aren't reported as gaps.
    Returns a list of issue dicts (geometries still in metres)."""
    issues = []
    geoms = [p["geometry"] for p in parcels]
    ids = [p["id"] for p in parcels]

    for pid, g in zip(ids, geoms):
        if not g.is_valid:
            issues.append(_issue("INVALID", "high", [pid], g,
                                 "Self-intersecting or malformed boundary.",
                                 "Rebuild with make_valid and keep the largest valid part."))
            g = make_valid(g)
        parts = _polygons(g)
        if len(parts) > 1:
            issues.append(_issue("MULTIPART", "medium", [pid], g,
                                 f"Parcel is split into {len(parts)} disconnected pieces.",
                                 "Keep the largest piece; merge the rest into their neighbours."))
        if g.area < MIN_PARCEL_M2:
            issues.append(_issue("TOO_SMALL", "medium", [pid], g,
                                 f"Area {g.area:.1f} m2 is below the {MIN_PARCEL_M2} m2 minimum plot size.",
                                 "Merge into the neighbour sharing the longest boundary."))
        elif g.area <= SLIVER_MAX_M2 and thinness(g) < SLIVER_THINNESS:
            issues.append(_issue("SLIVER", "medium", [pid], g,
                                 f"Long, thin shape (thinness {thinness(g):.2f}) -- typical digitising artefact.",
                                 "Merge into the neighbour sharing the longest boundary."))

    valid = [make_valid(g) if not g.is_valid else g for g in geoms]
    tree = STRtree(valid)
    for i, g in enumerate(valid):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            inter = g.intersection(valid[j])
            if inter.area <= OVERLAP_TOL_M2:
                continue
            smaller = min(g.area, valid[j].area)
            union_area = g.union(valid[j]).area
            if union_area > 0 and inter.area / union_area >= DUPLICATE_IOU:
                issues.append(_issue("DUPLICATE", "high", [ids[i], ids[j]], inter,
                                     "Two parcels have essentially the same boundary.",
                                     "Delete one of the two."))
            else:
                issues.append(_issue("OVERLAP", "high", [ids[i], ids[j]], inter,
                                     f"Parcels overlap by {inter.area:.1f} m2 ({100 * inter.area / smaller:.0f}% of the smaller).",
                                     "Assign the shared area to one parcel and cut it from the other."))

    union = unary_union(valid)
    for poly in _polygons(union):
        for ring in poly.interiors:
            hole = Polygon(ring)
            if exclude is not None:
                hole = hole.difference(exclude)
            if GAP_TOL_M2 < hole.area <= MAX_GAP_M2:
                neighbours = [ids[k] for k in tree.query(hole, predicate="touches")]
                issues.append(_issue("GAP", "medium", neighbours, hole,
                                     f"Unclaimed {hole.area:.1f} m2 gap enclosed by parcels.",
                                     "Absorb the gap into the neighbour sharing the longest boundary."))
    return issues


def _shared_length(a, b):
    return a.boundary.intersection(b.boundary).length


def _best_neighbour(geom, candidates):
    best, best_len = None, 0.0
    for idx, other in candidates:
        length = _shared_length(geom, other)
        if length > best_len:
            best, best_len = idx, length
    return best


def auto_fix(parcels, exclude=None):
    """Returns (fixed_parcels, change_log). Order: make valid -> drop
    duplicates -> resolve overlaps (larger/higher-confidence parcel keeps the
    shared area) -> merge slivers/too-small into best neighbour -> fill
    small enclosed gaps."""
    log = []
    work = []
    for p in parcels:
        g = p["geometry"]
        if not g.is_valid:
            parts = _polygons(make_valid(g))
            g = max(parts, key=lambda x: x.area) if parts else Polygon()
            log.append({"parcel_ids": [p["id"]], "action": "made valid"})
        elif g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
            log.append({"parcel_ids": [p["id"]], "action": "kept largest part"})
        work.append({**p, "geometry": g})

    def rank(p):
        return (p.get("confidence", 0.0), p["geometry"].area)

    # duplicates and overlaps
    tree = STRtree([p["geometry"] for p in work])
    dropped = set()
    for i, p in enumerate(work):
        if i in dropped:
            continue
        for j in tree.query(p["geometry"], predicate="intersects"):
            j = int(j)
            if j <= i or j in dropped:
                continue
            a, b = work[i]["geometry"], work[j]["geometry"]
            inter = a.intersection(b)
            if inter.area <= OVERLAP_TOL_M2:
                continue
            if inter.area / a.union(b).area >= DUPLICATE_IOU:
                loser = j if rank(work[i]) >= rank(work[j]) else i
                dropped.add(loser)
                log.append({"parcel_ids": [work[loser]["id"]], "action": "removed duplicate"})
                if loser == i:
                    break
                continue
            winner, loser = (i, j) if rank(work[i]) >= rank(work[j]) else (j, i)
            cut = _polygons(work[loser]["geometry"].difference(work[winner]["geometry"]))
            work[loser]["geometry"] = max(cut, key=lambda x: x.area) if cut else Polygon()
            log.append({"parcel_ids": [work[winner]["id"], work[loser]["id"]],
                        "action": f"overlap of {inter.area:.1f} m2 assigned to parcel {work[winner]['id']}"})
    work = [p for k, p in enumerate(work) if k not in dropped and not p["geometry"].is_empty]

    # slivers / too small -> merge into best neighbour
    changed = True
    while changed:
        changed = False
        tree = STRtree([p["geometry"] for p in work])
        for i, p in enumerate(work):
            g = p["geometry"]
            small = g.area < MIN_PARCEL_M2
            sliver = g.area <= SLIVER_MAX_M2 and thinness(g) < SLIVER_THINNESS
            if not (small or sliver):
                continue
            cands = [(int(j), work[int(j)]["geometry"]) for j in tree.query(g, predicate="intersects") if int(j) != i]
            target = _best_neighbour(g, cands)
            if target is None:
                continue
            merged = unary_union([work[target]["geometry"], g])
            parts = _polygons(merged)
            work[target]["geometry"] = max(parts, key=lambda x: x.area) if len(parts) > 1 else merged
            log.append({"parcel_ids": [p["id"], work[target]["id"]],
                        "action": f"{'too-small' if small else 'sliver'} parcel {p['id']} merged into {work[target]['id']}"})
            work.pop(i)
            changed = True
            break

    # enclosed gaps -> best neighbour
    tree = STRtree([p["geometry"] for p in work])
    for poly in _polygons(unary_union([p["geometry"] for p in work])):
        for ring in poly.interiors:
            hole = Polygon(ring)
            if exclude is not None:
                hole = hole.difference(exclude)
            if hole.geom_type != "Polygon" or not (GAP_TOL_M2 < hole.area <= MAX_GAP_M2):
                continue
            cands = [(int(j), work[int(j)]["geometry"]) for j in tree.query(hole, predicate="intersects")]
            target = _best_neighbour(hole, cands)
            if target is None:
                continue
            merged = unary_union([work[target]["geometry"], hole])
            if merged.geom_type == "Polygon":
                work[target]["geometry"] = merged
                log.append({"parcel_ids": [work[target]["id"]],
                            "action": f"filled {hole.area:.1f} m2 gap into parcel {work[target]['id']}"})

    return work, log


def summarize(issues):
    counts = {}
    for it in issues:
        counts[it["type"]] = counts.get(it["type"], 0) + 1
    return counts
