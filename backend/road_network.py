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
