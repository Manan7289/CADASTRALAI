"""Road network from the roads & lanes mask: centreline segments between junctions,
each with its length, median width and a width class.

The PS asks for roads, pathways and corridors; a surveyor wants them as a network
(which lane is 2.5 m wide, where it meets the main road), not only as one paved area.
"""
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MIN_SPUR_M = 6.0      # dead-end branches shorter than this are skeleton whiskers at the road edge, not lanes
SIMPLIFY_M = 1.0
WIDTH_CLASSES = [(3.0, "lane"), (8.0, "street"), (float("inf"), "main road")]   # upper bound (m), class


def width_class(w):
    return next(name for top, name in WIDTH_CLASSES if w < top)


def _trace(skel):
    """Pixel paths between nodes (endpoints and junctions) of an 8-connected skeleton."""
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    node = skel & (nb != 2)
    H, W = skel.shape
    seen = np.zeros_like(skel)
    paths = []

    def walk(start, first):
        path = [start, first]
        prev, cur = start, first
        while not node[cur]:
            seen[cur] = True
            nxt = None
            for dy, dx in NEIGH:
                q = (cur[0] + dy, cur[1] + dx)
                if 0 <= q[0] < H and 0 <= q[1] < W and skel[q] and q != prev and q not in path[-3:]:
                    if node[q] or not seen[q]:
                        nxt = q
                        break
            if nxt is None:
                break
            prev, cur = cur, nxt
            path.append(cur)
        return path

    for y, x in zip(*np.nonzero(node)):
        for dy, dx in NEIGH:
            q = (y + dy, x + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and skel[q] and not seen[q] and not node[q]:
                paths.append(walk((y, x), q))
    # closed loops with no node at all
    for y, x in zip(*np.nonzero(skel & ~seen & ~node)):
        if seen[y, x]:
            continue
        for dy, dx in NEIGH:
            q = (y + dy, x + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and skel[q] and not seen[q]:
                node[y, x] = True
                paths.append(walk((y, x), q))
                break
    return paths, node


def road_segments(skel, width, transform, gsd):
    """skel: bool centreline raster; width: road width (m) per pixel; transform: pixel -> metres."""
    paths, node = _trace(skel)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    segs = []
    for p in paths:
        if len(p) < 2:
            continue
        rows, cols = np.array(p).T
        xs, ys = transform * (cols + 0.5, rows + 0.5)
        line = LineString(np.column_stack([xs, ys]))
        dead_end = nb[p[0]] == 1 or nb[p[-1]] == 1
        w = float(np.median(width[rows, cols]))
        if dead_end and line.length < max(MIN_SPUR_M, w):
            continue
        line = line.simplify(SIMPLIFY_M)
        segs.append({"geometry": line, "length_m": round(line.length, 1), "width_m": round(w, 1),
                     "class": width_class(w)})
    for i, s in enumerate(sorted(segs, key=lambda s: -s["length_m"]), start=1):
        s["id"] = i
    return segs


BRIDGE_MAX_M = 30        # longest gap in a lane (tree canopy, parked trucks) that is closed
BRIDGE_DIR_M = 8         # how far back along the lane its direction is measured
BRIDGE_MIN_LANE_M = 15   # only a real lane (at least this long) is extended; short paved bits are not
LANE_MIN_W_M, LANE_MAX_W_M = 3.0, 8.0


def _walk_back(skel, p, steps):
    """Follow the skeleton from endpoint p for up to `steps` pixels; returns the last point reached."""
    prev, cur = None, p
    H, W = skel.shape
    for _ in range(steps):
        nxt = None
        for dy, dx in NEIGH:
            q = (cur[0] + dy, cur[1] + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and skel[q] and q != prev:
                nxt = q
                break
        if nxt is None:
            break
        prev, cur = cur, nxt
    return cur


def bridge_lane_gaps(corridors, blocked, gsd):
    """Close gaps in the lane network: a lane that dead-ends is extended straight ahead, and if
    it reaches another road within BRIDGE_MAX_M without crossing a building (`blocked`), the gap
    becomes lane. Without this, a tree over a lane joins two blocks into one and every parcel
    in them is laid out wrong."""
    import cv2
    from skimage.morphology import skeletonize
    skel = skeletonize(corridors)
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    comp, _ = ndi.label(corridors)
    width = ndi.distance_transform_edt(corridors) * 2 * gsd
    skel_len = ndi.sum(skel, comp, index=np.arange(comp.max() + 1)) * gsd
    out = corridors.copy()
    H, W = corridors.shape
    n_bridges = 0
    for y, x in zip(*np.nonzero(skel & (nb == 1))):
        by, bx = _walk_back(skel, (y, x), int(BRIDGE_DIR_M / gsd))
        d = np.array([y - by, x - bx], float)
        if np.hypot(*d) < 3 / gsd:
            continue
        d /= np.hypot(*d)
        w = float(np.clip(width[y, x] * 1.6, LANE_MIN_W_M, LANE_MAX_W_M))
        own = comp[y, x]
        if skel_len[own] < BRIDGE_MIN_LANE_M:
            continue
        start = w / gsd                      # skip the lane's own rounded end
        hit = None
        for t in np.arange(start, BRIDGE_MAX_M / gsd, 1.0):
            py, px = int(round(y + d[0] * t)), int(round(x + d[1] * t))
            if not (0 <= py < H and 0 <= px < W) or blocked[py, px]:
                break
            if corridors[py, px] and (comp[py, px] != own or t * gsd > 2 * BRIDGE_DIR_M):
                hit = (py, px)
                break
        if hit is None:
            continue
        line = np.zeros((H, W), np.uint8)
        cv2.line(line, (int(x), int(y)), (hit[1], hit[0]), 1, thickness=max(1, int(round(w / gsd))))
        out |= line.astype(bool) & ~blocked
        n_bridges += 1
    return out, n_bridges
