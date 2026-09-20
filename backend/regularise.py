"""Give parcels survey-like geometry: straight edges, shared lines, no fingers.

The watershed draws a parcel boundary pixel by pixel, so an edge that should be one straight
wall comes out with twenty kinks, and a plot can send a thin finger down a lane. Real cadastral
plots are a few straight segments, neighbours share one exact line, and a row of plots lines up
with its street.

So the parcel coverage is straightened as a whole, never polygon by polygon (that would open
gaps between neighbours): the shared boundary network is split into arcs between junctions, each
arc is straightened once, and the polygons are rebuilt from the straightened arcs. Every arc
belongs to both its neighbours, so they keep agreeing exactly. Each arc takes its grid direction
from the nearest building, so a row of houses sets the direction of the plot lines around it.

Plot sizes are left alone: in an organically grown neighbourhood they really do differ, and
forcing them equal would stop them matching the walls in the image.
"""
import math

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

SIMPLIFY_M = 1.2          # drop pixel wobble before straightening
CHORD_TOL_M = 2.0         # an arc that never leaves this band becomes one straight line
ANGLE_TOL_DEG = 25        # a segment this close to the local grid is snapped onto it
MIN_SEG_M = 1.5           # shorter segments are absorbed rather than snapped
CUT_TOL_M = 0.5           # a straightened line may not run this much further through a house


def regularise(parcels, buildings, exclude=None, keep_area_ratio=0.5):
    """parcels, buildings: {id: Polygon} in metres. Returns {id: Polygon}, same ids.

    Falls back to the input whenever the rebuild loses area or ids, so a bad case degrades to
    today's output rather than to a broken layer."""
    ids = [i for i, g in parcels.items() if g is not None and not g.is_empty]
    if len(ids) < 2:
        return parcels
    before = unary_union([parcels[i] for i in ids])
    grid = _grid_directions(buildings)
    bl = [g for g in buildings.values() if g is not None and not g.is_empty]
    btree = STRtree(bl) if bl else None

    arcs = _arcs([parcels[i] for i in ids])
    straight = []
    for arc in arcs:
        s = _straighten(arc, grid, btree, bl)
        if s is not None and s.length > 0:
            straight.append(s)
    if not straight:
        return parcels

    rebuilt = [p for p in polygonize(unary_union(straight)) if p.is_valid and not p.is_empty]
    if not rebuilt:
        return parcels
    out = _reassign(rebuilt, parcels, ids)
    out = _restore(out, before)             # straightening can shave the outer edge: give the land back
    out = _fill_orphan_holes(out)           # no parcel should wrap around land nobody claims
    for _ in range(3):                      # parcels the validator complains about keep their old
        bad = _bad_ids(out, exclude)         # shape; the rest stay straightened
        if not bad:
            break
        out = _revert(out, parcels, bad)
        out = _restore(out, before)
    out = _restore(out, before)
    out = _clip(out, before)                # straightened lines may bulge past the survey edge
    after = unary_union([g for g in out.values() if g is not None and not g.is_empty])
    if (len(out) < len(ids) or after.area < before.area * keep_area_ratio
            or len(_issues(out, exclude)) > len(_issues(parcels, exclude))):
        return parcels                      # rebuild lost parcels, land or tidiness: keep the old geometry
    return {i: (out.get(i) or parcels[i]) for i in parcels}


def _grid_directions(buildings):
    """(centroid, angle) per building: the direction its walls run, from its bounding rectangle."""
    pts, angs = [], []
    for g in buildings.values():
        if g is None or g.is_empty or g.area <= 0:
            continue
        r = g.minimum_rotated_rectangle
        c = list(r.exterior.coords)
        if len(c) < 3:
            continue
        (x0, y0), (x1, y1), (x2, y2) = c[0], c[1], c[2]
        e1, e2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        dx, dy = (x1 - x0, y1 - y0) if e1 >= e2 else (x2 - x1, y2 - y1)
        pts.append(g.centroid); angs.append(math.atan2(dy, dx) % math.pi)
    if not pts:
        return None
    return STRtree(pts), np.array(angs), pts


def _local_angle(grid, point):
    if grid is None:
        return None
    tree, angs, pts = grid
    j = tree.nearest(point)
    return float(angs[j]) if j is not None else None


def _arcs(polys):
    """Split the shared boundary network into arcs that run between junctions."""
    lines = unary_union([p.boundary for p in polys])          # noded at every crossing
    merged = linemerge(lines)                                  # merges only through degree-2 nodes
    parts = list(merged.geoms) if isinstance(merged, MultiLineString) else [merged]
    return [p for p in parts if isinstance(p, LineString) and p.length > 0]


def _straighten(arc, grid, btree=None, bl=None):
    """One straight line when the arc barely bends; otherwise grid-snapped segments.
    A candidate that would run further through a house than the original is refused: a plot line
    cutting a building is worse than a line with a kink in it."""
    simple = arc.simplify(SIMPLIFY_M, preserve_topology=False)
    cs = list(simple.coords)
    if len(cs) < 3:
        return simple
    closed = cs[0] == cs[-1]
    if not closed:
        chord = LineString([cs[0], cs[-1]])
        if chord.length > 0 and max(Point(c).distance(chord) for c in cs[1:-1]) <= CHORD_TOL_M:
            if not _cuts_house(chord, arc, btree, bl):
                return chord
    theta = _local_angle(grid, simple.interpolate(0.5, normalized=True))
    if theta is None:
        return simple
    snapped = _snap(cs, theta, closed)
    if _cuts_house(snapped, arc, btree, bl):
        return simple if not _cuts_house(simple, arc, btree, bl) else arc
    return snapped


def _cuts_house(cand, original, btree, bl):
    """True when cand runs through a building noticeably more than the original line did."""
    if btree is None or cand.is_empty:
        return False
    for j in btree.query(cand):
        b = bl[j]
        if cand.intersection(b).length > original.intersection(b).length + CUT_TOL_M:
            return True
    return False


def _snap(cs, theta, closed):
    """Snap each segment onto the local grid, then put the corners back where the
    snapped lines cross, so the arc stays connected and its ends stay put."""
    dirs, mids = [], []
    for a, b in zip(cs, cs[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        if seg >= MIN_SEG_M:
            for cand in (theta, theta + math.pi / 2):
                for k in (0, math.pi):
                    d = (ang - (cand + k) + math.pi) % (2 * math.pi) - math.pi
                    if abs(math.degrees(d)) <= ANGLE_TOL_DEG:
                        ang = cand + k
                        break
        dirs.append(ang); mids.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))

    out = [cs[0]]
    for k in range(len(dirs) - 1):
        p = _intersect(mids[k], dirs[k], mids[k + 1], dirs[k + 1])
        out.append(p if p is not None else cs[k + 1])
    out.append(cs[-1] if not closed else out[0])
    line = LineString(out)
    return line if line.is_valid and line.length > 0 else LineString(cs)


def _intersect(p1, a1, p2, a2):
    d1, d2 = (math.cos(a1), math.sin(a1)), (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _restore(out, before):
    """Land inside the old coverage that no rebuilt parcel claims goes to the parcel it shares
    the longest edge with, so the layer stays gap-free."""
    ids = [i for i, g in out.items() if g is not None and not g.is_empty]
    if not ids:
        return out
    after = unary_union([out[i] for i in ids])
    left = before.difference(after)
    if left.is_empty:
        return out
    pieces = list(left.geoms) if left.geom_type.startswith("Multi") else [left]
    tree = STRtree([out[i] for i in ids])
    for piece in pieces:
        if piece.is_empty or piece.area <= 0:
            continue
        best, best_len = None, 0.0
        for j in tree.query(piece.buffer(0.05)):
            shared = out[ids[j]].buffer(0.05).intersection(piece).area
            if shared > best_len:
                best, best_len = ids[j], shared
        if best is None:
            j = tree.nearest(piece.representative_point())
            best = ids[j] if j is not None else None
        if best is not None:
            merged = unary_union([out[best], piece])
            if merged.geom_type == "MultiPolygon":
                merged = max(merged.geoms, key=lambda p: p.area)
            out[best] = merged
    return out


def _issues(polys, exclude):
    """The app's own topology checks — overlaps, gaps, holes, slivers — so the straightened layer
    is judged by exactly the rules the survey is judged by."""
    import topology
    items = [{"id": i, "geometry": g} for i, g in polys.items() if g is not None and not g.is_empty]
    try:
        return topology.validate(items, exclude=exclude)
    except Exception:
        return []


def _clip(out, before):
    """Keep every parcel inside the area the survey actually covers."""
    for i, g in list(out.items()):
        if g is None or g.is_empty or before.contains(g):
            continue
        c = g.intersection(before)
        if c.is_empty:
            continue
        if c.geom_type == "MultiPolygon":
            c = max(c.geoms, key=lambda p: p.area)
        if c.geom_type == "Polygon":
            out[i] = c
    return out


def _revert(out, parcels, bad):
    """Put a parcel back to its old shape, trimmed so it cannot overlap the straightened
    neighbours it now sits against."""
    keep = [i for i, g in out.items() if g is not None and not g.is_empty and i not in bad]
    tree = STRtree([out[i] for i in keep]) if keep else None
    for i in bad:
        g = parcels.get(i)
        if g is None or g.is_empty:
            continue
        if tree is not None:
            others = [out[keep[j]] for j in tree.query(g) if out[keep[j]].intersects(g)]
            if others:
                g = g.difference(unary_union(others))
        if g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        out[i] = g
    return out


def _bad_ids(polys, exclude=None):
    """Parcels named in a topology issue the straightened layer has."""
    bad = set()
    for iss in _issues(polys, exclude):
        bad.update(iss.get("parcel_ids") or [])
    return bad


def _fill_orphan_holes(out):
    """An enclosed ring that no other parcel occupies is this parcel's own land, so fill it."""
    ids = [i for i, g in out.items() if g is not None and g.geom_type == "Polygon" and g.interiors]
    if not ids:
        return out
    keys = [i for i, g in out.items() if g is not None and not g.is_empty]
    tree = STRtree([out[i] for i in keys])
    for i in ids:
        g = out[i]
        keep = []
        for ring in g.interiors:
            hole = Polygon(ring)
            if hole.area <= 0:
                continue
            taken = 0.0
            for j in tree.query(hole):
                if keys[j] != i:
                    taken += hole.intersection(out[keys[j]]).area
            if taken > hole.area * 0.1:
                keep.append(ring)                  # a neighbour really is in there: leave it
        if len(keep) != len(g.interiors):
            out[i] = Polygon(g.exterior, keep)
    return out


def _reassign(rebuilt, parcels, ids):
    """Each rebuilt face goes to the parcel it covers most; a parcel's faces are merged."""
    tree = STRtree([parcels[i] for i in ids])
    groups = {}
    for face in rebuilt:
        c = face.representative_point()
        best, best_area = None, 0.0
        for j in tree.query(face):
            inter = face.intersection(parcels[ids[j]]).area
            if inter > best_area:
                best, best_area = ids[j], inter
        if best is None:
            j = tree.nearest(c)
            best = ids[j] if j is not None else None
        if best is not None:
            groups.setdefault(best, []).append(face)
    out = {}
    for i, faces in groups.items():
        g = unary_union(faces)
        if g.geom_type == "MultiPolygon":                      # keep the main piece only
            g = max(g.geoms, key=lambda p: p.area)
        out[i] = g
    return out


