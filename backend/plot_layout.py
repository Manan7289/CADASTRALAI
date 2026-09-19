"""Parcels laid out the way planned Indian colonies are surveyed.

A colony is a grid of blocks bounded by roads. Each block holds one row of plots, or
two rows back to back with a rear line through the middle. Every plot fronts a road,
its side boundaries run straight back from the road, and one house stands on it with
its setback, courtyard and boundary wall.

So parcels are built in that order, each step on top of an AI layer:

    block      land enclosed by the AI road & lane network
    rows       the block is cut along its long axis where no house stands
               (the rear line), if it is deep enough for two rows
    plots      each row is cut across, between neighbouring houses from the AI roof
               layer, so every house gets its own plot
    walls      each cut is moved to the strongest visible edge (a compound wall,
               a fence) in the gap between the two houses
    vacant     a long stretch with no house is split into plots of the typical
               width of that row; a large empty block (park, ground) stays whole

Cuts are straight lines in the block's own frame, so plots come out as clean
quadrilaterals, not blobs.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

MIN_BLOCK_M2 = 30
MIN_PLOT_M2 = 12            # same minimum as the topology checker
MAX_ROW_DEPTH_M = 34        # a row deeper than this is split again at the next rear line
MIN_ROW_DEPTH_M = 8         # never cut a row shallower than this
REAR_LINE_MAX_OCC = 0.3     # a rear line may cross at most this share of roofs along its length
SNAP_M = 1.5                # how far a cut may move to reach a wall line
PLOT_WIDTH_DEFAULT_M = 12   # typical plot frontage when a row has too few houses to measure it
PLOT_WIDTH_RANGE_M = (6, 25)
VACANT_FACTOR = 1.7         # a gap wider than this many typical plots holds vacant plots of its own
OPEN_SPACE_M2 = 2500        # a house-less block bigger than this is one open-space parcel (park, ground)
SIDE_MARGIN_M = 1.5         # land a house keeps beside it when the next plot is vacant
ROOF_KEEP_SHARE = 0.85      # a house split across plots merges them unless one plot holds this much of it
MERGE_MAX_ROOF_M2 = 250     # ...but only a house-sized roof; a bigger one is several attached houses
FRONT_STRAIGHTEN_M = 2.5    # a plot front is a straight line; land up to this far beyond it is road verge
FRONT_TAKE_MAX_M = 1.0      # ...and it may step at most this far into the road (never the lane's middle)
VERGE_MAX_DEPTH_M = 8       # a house-less strip thinner than this is a road median / verge, not a plot
OPEN_STRIP_MAX_DEPTH_M = 20  # a house-less block thinner than this (green median, road reserve) stays one parcel
# L / T shaped blocks are split at their inner corners into rectangular parts
RECT_OK = 0.8                # a block filling this share of its bounding rectangle is laid out as it is
MIN_DEFECT_M = 6             # an inner corner at least this deep is a place to split
MIN_PART_M2 = 400
# block types (how a surveyor would treat the block)
KIND_CODE = {"plotted": 1, "campus": 2, "sparse": 3, "organic": 4}
KIND_LABEL = {1: "Plotted colony (plots in rows)", 2: "Walled compound (one parcel)",
              3: "Scattered buildings in open land", 4: "Unplanned (plots follow the houses)"}
BIG_BUILDING_M2 = 800        # a roof this big and not long-and-thin is one building (apartment, school, market);
                             # smaller "big" roofs are usually attached houses detected as one blob
ROW_ASPECT = 2.5             # ...a long thin roof is more likely a row of attached houses detected as one
CAMPUS_MIN_BUILDING_M2 = 600 # a block dominated by a building this big is one walled compound
CAMPUS_MAX_HOUSES = 3
ORGANIC_MIN_ROOFS = 5
PLOTTED_MIN_COVER = 0.22     # a plotted colony is dense: houses cover at least this share of the block
PLOTTED_MIN_HOUSES = 4
NEIGHBOUR_M = 3.0            # in a plotted colony most houses stand within this of the next house...
PLOTTED_MIN_NEIGHBOURED = 0.5  # ...at least this share of them (society towers stand far apart)
LANE_FLANK_M = 8             # a hidden lane has houses within this distance on both sides...
LANE_FLANK_MIN = 0.15        # ...covering at least this share of each side band
ALIGN_DEG = 12               # a roof within this angle of the block axis is "in line"
ORGANIC_MAX_ALIGNED = 0.5    # fewer roofs in line than this: an unplanned block, parcels follow the houses
STANDARD_FRONTAGE_M = (9.1, 12.2, 15.2, 18.3)   # 30, 40, 50, 60 ft sites
PLOT_MIN_W_M = 4.5           # no plot narrower than this
PLOT_MAX_FACTOR = 2.2        # nor wider than this many typical plots (a big building excepted)
ROOF_CUT_COST = 12.0         # cost of a cut crossing a house-sized roof (x share of the row depth that is roof)
BLOB_CUT_COST = 2.5          # ...crossing a blob of attached houses detected as one: cheap, cut on its party walls
WALL_BONUS = 0.6            # extra cost of a cut where no wall line is visible
WIDTH_COST = 3.0             # cost of a plot 2x (or 0x) the typical width
OPEN_PLOT_COST = 0.3         # cost of an open-land parcel (no building) in a sparse row
DENSE_ROW_COVER = 0.45       # a row whose length is at least this share roofed is a colony row
# hidden streets: a block this big is several blocks whose separating lanes the imagery does not show
# (tree canopy); a roof-free stripe across it, as wide as a street plus two front yards, is a street
BIG_BLOCK_M2 = 2500
HIDDEN_STREET_LEN_M = 30
STRIPE_W_M = (5, 25)
STRIPE_MAX_OCC = 0.03        # share of the stripe covered by roofs
STRIPE_MIN_SPAN = 0.8        # the stripe must run across this share of the block
FRONT_YARD_M = 1.5           # land on each side of the hidden street that stays with the plots
HIDDEN_STREET_W_M = (3, 9)


def _block_angle(mask):
    """Angle (degrees) that turns the block's long axis horizontal."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), a = cv2.minAreaRect(np.vstack(cs))
    return a if w >= h else a + 90


def _rotate(img, M, size, nearest):
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _rot_frame(block):
    h, w = block.shape
    ang = _block_angle(block)
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos) + 4, int(h * cos + w * sin) + 4
    M[0, 2] += W / 2 - c[0]
    M[1, 2] += H / 2 - c[1]
    return M, (W, H)


def _stripe_runs(occ, present, gsd, extent_ok):
    """Runs of rows with (almost) no roof, as wide as a street with front yards, not at the block edge."""
    idx = np.nonzero(present)[0]
    if idx.size == 0:
        return []
    lo_edge, hi_edge = idx[0] + 10 / gsd, idx[-1] - 10 / gsd
    free = (occ <= STRIPE_MAX_OCC) & present & extent_ok
    runs, start = [], None
    for i, f in enumerate(np.append(free, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            wm = (i - start) * gsd
            if STRIPE_W_M[0] <= wm <= STRIPE_W_M[1] and start > lo_edge and i < hi_edge:
                runs.append((start, i))
            start = None
    return runs


def _open(mask, kh, kw):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kw)), max(1, int(kh))))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k) > 0


def _erode(mask, kh, kw):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kw)), max(1, int(kh))))
    return cv2.erode(mask.astype(np.uint8), k) > 0


def hidden_streets(block, roofs, gsd):
    """Street mask (block crop): long roof-free corridors across the block, in both grid directions.

    Rows of houses face each other across a lane; seen from above that lane is a long stripe with
    no roof, as wide as the road plus two front yards. Where the imagery does not show the lane
    itself (tree canopy), this stripe does. A corridor must be at least HIDDEN_STREET_LEN_M long;
    the middle of it (minus the front yards, which stay with the plots) becomes street. Open areas
    much wider than a street (a park, a ground) are not streets."""
    M, (W, H) = _rot_frame(block)
    rb = _rotate(block.astype(np.uint8), M, (W, H), True) > 0
    rr = _rotate((roofs > 0).astype(np.uint8), M, (W, H), True) > 0
    free = rb & ~(cv2.dilate(rr.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, int(0.5 / gsd))) > 0)
    L, Wmin = int(HIDDEN_STREET_LEN_M / gsd), int(STRIPE_W_M[0] / gsd)
    yard = int(FRONT_YARD_M / gsd)
    sw = HIDDEN_STREET_W_M[0] / gsd
    wide = _open(free, STRIPE_W_M[1] / gsd, STRIPE_W_M[1] / gsd)
    st = np.zeros_like(rb)
    for horizontal in (True, False):
        corr = _open(free, Wmin, L) if horizontal else _open(free, L, Wmin)
        # keep the middle of the corridor: drop a front yard's width from both sides
        mid = _erode(corr, 2 * yard + 1, 1) if horizontal else _erode(corr, 1, 2 * yard + 1)
        # at least a street's width left after the yards, and still long
        mid = _open(mid, sw, 1) if horizontal else _open(mid, 1, sw)
        mid = _open(mid, 1, L) if horizontal else _open(mid, L, 1)
        st |= mid
    st &= ~(cv2.dilate(wide.astype(np.uint8), np.ones((2 * yard + 1, 2 * yard + 1), np.uint8)) > 0)
    # a lane runs between two rows of houses: keep a candidate only with houses along both sides
    comps, n = ndi.label(st)
    fl = int(LANE_FLANK_M / gsd)
    for c, csl in enumerate(ndi.find_objects(comps), start=1):
        if csl is None:
            continue
        hgt, wid = csl[0].stop - csl[0].start, csl[1].stop - csl[1].start
        horizontal = wid >= hgt
        if horizontal:
            above = rr[max(0, csl[0].start - fl):csl[0].start, csl[1]]
            below = rr[csl[0].stop:csl[0].stop + fl, csl[1]]
        else:
            above = rr[csl[0], max(0, csl[1].start - fl):csl[1].start]
            below = rr[csl[0], csl[1].stop:csl[1].stop + fl]
        if min(above.mean() if above.size else 0, below.mean() if below.size else 0) < LANE_FLANK_MIN:
            st[csl][comps[csl] == c] = False
    if not st.any():
        return np.zeros_like(block)
    Minv = cv2.invertAffineTransform(M)
    return (_rotate(st.astype(np.uint8), Minv, block.shape[::-1], True) > 0) & block


def _roof_shapes(roofs, gsd):
    """id -> (area m2, angle deg, aspect) for each roof in a block crop."""
    out = {}
    for rid, sl in enumerate(ndi.find_objects(roofs), start=1):
        if sl is None:
            continue
        m = (roofs[sl] == rid).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        (_, _), (w, h), a = cv2.minAreaRect(np.vstack(cs))
        out[rid] = (m.sum() * gsd * gsd, a if w >= h else a + 90, max(w, h) / max(min(w, h), 1))
    return out


def _neighboured_share(roofs, gsd):
    """Share of roofs with another roof within NEIGHBOUR_M."""
    from skimage.segmentation import expand_labels
    grown = expand_labels(roofs, distance=NEIGHBOUR_M / gsd / 2)
    ids = np.unique(roofs[roofs > 0])
    if ids.size == 0:
        return 0.0
    # pairs of different labels meeting horizontally or vertically
    a = np.concatenate([grown[:, :-1].ravel(), grown[:-1, :].ravel()])
    b = np.concatenate([grown[:, 1:].ravel(), grown[1:, :].ravel()])
    touch = (a != b) & (a > 0) & (b > 0)
    neighboured = np.union1d(a[touch], b[touch])
    return float(np.isin(ids, neighboured).mean())


def block_type(block, roofs, gsd):
    """'campus' (one walled compound), 'sparse' (scattered buildings in open land), 'organic'
    (unplanned: houses at random angles) or 'plotted' (a colony of plots in rows)."""
    shapes = _roof_shapes(roofs, gsd)
    if not shapes:
        return "plotted"
    big = [v for v in shapes.values() if v[0] >= CAMPUS_MIN_BUILDING_M2]
    houses = [v for v in shapes.values() if v[0] < MERGE_MAX_ROOF_M2]
    if big and len(houses) <= CAMPUS_MAX_HOUSES:
        return "campus"
    cover = (roofs > 0).sum() / max(block.sum(), 1)
    if len(houses) < PLOTTED_MIN_HOUSES or cover < PLOTTED_MIN_COVER or _neighboured_share(roofs, gsd) < PLOTTED_MIN_NEIGHBOURED:
        return "sparse"            # scattered buildings (apartment towers in a society, bungalows): no plot grid
    if len(shapes) >= ORGANIC_MIN_ROOFS:
        axis = _block_angle(block)
        dirs = [v for v in shapes.values() if v[2] > 1.15]       # square roofs have no clear direction
        if len(dirs) >= ORGANIC_MIN_ROOFS:
            dev = [min(abs((v[1] - axis) % 90), 90 - abs((v[1] - axis) % 90)) for v in dirs]
            if np.mean(np.array(dev) <= ALIGN_DEG) < ORGANIC_MAX_ALIGNED:
                return "organic"
    return "plotted"


def _row_cuts(x0, x1, typ, roof_prof, edge_prof, forbid, gsd, step_m=0.5, roof_cols=None, dense=True):
    """Cut positions across one row, chosen together by dynamic programming.

    Each cut costs the roof it crosses (heavily: a house should not be split) minus the wall
    evidence under it; each plot costs how far its width is from the row's typical frontage.
    `forbid` marks columns inside one big building, which may not be cut at all.
    In a sparse row (dense=False) a plot with no roof in it is open land and may be any width:
    open land between scattered buildings is not cut into plots that do not exist."""
    step = max(1, int(round(step_m / gsd)))
    xs = np.arange(x0, x1, step)
    xs = np.append(xs, x1)
    n = len(xs)
    wmin, wmax = PLOT_MIN_W_M / gsd, PLOT_MAX_FACTOR * typ
    xi = np.clip(xs, 0, len(roof_prof) - 1)
    emax = max(float(edge_prof[x0:x1].max()) if x1 > x0 else 0.0, 1e-6)
    # a wall line makes a cut cheaper but never free: cuts are only made where a plot is needed
    cut_cost = ROOF_CUT_COST * roof_prof[xi] + WALL_BONUS * (1 - edge_prof[xi] / emax)
    cut_cost[forbid[xi]] = np.inf
    cut_cost[0] = cut_cost[-1] = 0.0
    roof_cum = None if roof_cols is None else np.concatenate([[0], np.cumsum(roof_cols[xi[1:]] > 0)])
    best = np.full(n, np.inf)
    prev = np.full(n, -1)
    best[0] = 0.0
    for j in range(1, n):
        if not np.isfinite(cut_cost[j]):
            continue
        w = xs[j] - xs[:j]
        width_cost = WIDTH_COST * ((w - typ) / typ) ** 2
        ok = (w >= wmin) & (w <= wmax)
        if not dense and roof_cum is not None:
            empty = roof_cum[j] - roof_cum[:j] == 0
            width_cost = np.where(empty, OPEN_PLOT_COST, width_cost)
            ok |= empty & (w >= wmin)
        cand = best[:j] + width_cost + cut_cost[j]
        if not (ok & np.isfinite(cand)).any():
            ok = w >= wmin                    # a big building may need a plot wider than allowed
        cand = np.where(ok, cand, np.inf)
        i = int(np.argmin(cand))
        if np.isfinite(cand[i]):
            best[j], prev[j] = cand[i], i
    if not np.isfinite(best[-1]):
        return []
    cuts, j = [], n - 1
    while prev[j] > 0:
        j = prev[j]
        cuts.append(float(xs[j]))
    return sorted(cuts)


def _row_angle(block, roofs, gsd):
    """Direction the rows of houses run in. Of the block's two grid directions, the one across
    which the roof profile has the clearest gaps (lanes, rear lines between rows)."""
    base = _block_angle(block)
    best, best_score = base, -1.0
    for ang in (base, base + 90):
        M, (W, H) = _rot_frame_at(block, ang)
        rb = _rotate(block.astype(np.uint8), M, (W, H), True) > 0
        rr = _rotate((roofs > 0).astype(np.uint8), M, (W, H), True) > 0
        n = rb.sum(1)
        occ = ndi.uniform_filter1d((rr & rb).sum(1) / np.maximum(n, 1), max(1, int(1 / gsd)))
        inside = n > 0.3 * max(n.max(), 1)
        # share of the block's depth lying in gaps between rows, but only gaps that have houses on both sides
        o = occ[inside]
        if o.size < 3:
            continue
        low = o < 0.08
        ys = np.nonzero(o >= 0.25)[0]
        if ys.size < 2:
            continue
        between = np.zeros_like(low)
        between[ys[0]:ys[-1]] = True
        score = (low & between).mean()
        if score > best_score + 0.02:
            best, best_score = ang, score
    return best


def _rot_frame_at(block, ang):
    h, w = block.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos) + 4, int(h * cos + w * sin) + 4
    M[0, 2] += W / 2 - c[0]
    M[1, 2] += H / 2 - c[1]
    return M, (W, H)


def _layout_block(block, roofs, edges, gsd, typ_global):
    """block: bool crop; roofs: int crop (roof ids); edges: float crop. Returns int crop of plot ids 1..n."""
    h, w = block.shape
    ang = _row_angle(block, roofs, gsd)
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(h * sin + w * cos) + 4, int(h * cos + w * sin) + 4
    M[0, 2] += W / 2 - c[0]
    M[1, 2] += H / 2 - c[1]
    rb = _rotate(block.astype(np.uint8), M, (W, H), True) > 0
    rr = _rotate(roofs.astype(np.int32).astype(np.float32), M, (W, H), True).astype(np.int32) * rb
    re = _rotate(edges.astype(np.float32), M, (W, H), False) * rb

    ys = np.nonzero(rb.any(1))[0]
    if ys.size == 0:
        return np.zeros_like(roofs, np.int32), np.zeros_like(block)
    y0, y1 = ys[0], ys[-1] + 1
    roof_ids = [i for i in np.unique(rr) if i > 0]
    boxes = {}
    for i in roof_ids:
        yy, xx = np.nonzero(rr == i)
        boxes[i] = (xx.min(), xx.max() + 1, yy.min(), yy.max() + 1, yy.mean())

    shapes = _roof_shapes(roofs, gsd)
    whole = {i for i, v in shapes.items() if v[0] >= BIG_BUILDING_M2 and v[2] < ROW_ASPECT}

    # rows: cut along the long axis at rear lines (where few houses stand), recursively
    # a rear line only "crosses" a house where it would go through its middle; the line where the
    # back walls of two back-to-back houses meet is free
    inner_y = np.zeros(rr.shape, bool)
    inner_y[1:-1] = (rr[:-2] == rr[2:]) & (rr[1:-1] == rr[:-2]) & (rr[1:-1] > 0)
    occ_row = inner_y.sum(1) / np.maximum(rb.sum(1), 1)
    min_d, max_d = MIN_ROW_DEPTH_M / gsd, MAX_ROW_DEPTH_M / gsd

    def split(a, b):
        depth = b - a
        if depth < 2 * min_d:
            return [(a, b)]
        lo, hi = int(a + min_d), int(b - min_d)
        prof = ndi.uniform_filter1d(occ_row, max(1, int(1.0 / gsd)))[lo:hi]
        # prefer a rear line near the middle: small penalty for distance from it
        pen = prof + 0.15 * np.abs(np.arange(lo, hi) - (a + b) / 2) / max(depth / 2, 1)
        k = lo + int(np.argmin(pen))
        houses_both_sides = any(bx[4] < k for bx in boxes.values()) and any(bx[4] > k for bx in boxes.values())
        if occ_row[k] <= REAR_LINE_MAX_OCC and (depth > max_d or (houses_both_sides and depth > 1.6 * min_d * 2)):
            return split(a, k) + split(k, b)
        return [(a, b)]

    rows = split(y0, y1)
    # a strip with no house (the yards between two rows) is not a row of its own: it joins the
    # neighbouring row, unless it is big enough to be open land in its own right
    def has_house(r):
        return any(r[0] <= bx[4] < r[1] for bx in boxes.values())
    if boxes and len(rows) > 1:
        merged = []
        for r in rows:
            if merged and (not has_house(r) or not has_house(merged[-1])) and \
                    (rb[r[0]:r[1]].sum() * gsd * gsd < OPEN_SPACE_M2 or not has_house(merged[-1])):
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        if len(merged) > 1 and not has_house(merged[0]):
            merged[1] = (merged[0][0], merged[1][1])
            merged = merged[1:]
        rows = merged

    # typical plot width: from the houses of this block, else the survey's
    # typical plot width: the narrower half of the houses (blobs of attached houses inflate a median)
    widths = [bx[1] - bx[0] for bx in boxes.values() if (bx[1] - bx[0]) * gsd <= PLOT_WIDTH_RANGE_M[1]]
    typ = np.clip(np.percentile(widths, 35) * gsd + SIDE_MARGIN_M if len(widths) >= 3 else typ_global, 6, 18) / gsd

    out = np.zeros((H, W), np.int32)
    n = 0
    block_area_m2 = rb.sum() * gsd * gsd
    for a, b in rows:
        band = np.zeros_like(rb)
        band[a:b] = rb[a:b]
        xs = np.nonzero(band.any(0))[0]
        if xs.size == 0:
            continue
        x0, x1 = xs[0], xs[-1] + 1
        in_row = [i for i, bx in boxes.items() if a <= bx[4] < b]
        if not boxes and (y1 - y0) * gsd < VERGE_MAX_DEPTH_M:
            continue                                    # road median / verge: left to the road
        if not boxes:
            cuts = []                                   # no house in the whole block: open land / ground / reserve, one parcel
        elif not in_row and band.sum() * gsd * gsd > OPEN_SPACE_M2 and block_area_m2 > OPEN_SPACE_M2:
            cuts = []                                   # park / ground / institutional open land: one parcel
        else:
            n_px = np.maximum(band[a:b].sum(0), 1)
            edge_prof = ndi.uniform_filter1d(re[a:b].sum(0) / n_px, 3)
            # a cut only costs where it goes through the middle of one house: the line where one
            # detected house ends and the touching next one begins (the party wall) is free
            seg = rr[a:b]
            inner = np.zeros(seg.shape, bool)
            inner[:, 1:-1] = (seg[:, :-2] == seg[:, 2:]) & (seg[:, 1:-1] == seg[:, :-2]) & (seg[:, 1:-1] > 0)
            house_px = inner & np.isin(seg, [i for i in in_row if shapes.get(i, (0,))[0] <= MERGE_MAX_ROOF_M2])
            blob_px = inner & ~house_px
            roof_prof = (house_px.sum(0) + blob_px.sum(0) * BLOB_CUT_COST / ROOF_CUT_COST) / n_px
            forbid = np.zeros(W, bool)
            for i in in_row:
                if i in whole:
                    forbid[boxes[i][0] + 1:boxes[i][1] - 1] = True
            # vacant stretches are cut at the standard site width nearest the row's typical plot
            std = min(STANDARD_FRONTAGE_M, key=lambda f: abs(f - typ * gsd)) / gsd
            roof_cols = (rr[a:b] > 0).any(0)
            dense = roof_cols[x0:x1].mean() >= DENSE_ROW_COVER
            cuts = _row_cuts(x0, x1, typ if in_row else std, roof_prof, edge_prof, forbid, gsd,
                             roof_cols=roof_cols, dense=dense)
        bounds = [x0] + [int(round(cx)) for cx in cuts if x0 < cx < x1] + [x1]
        for l, r in zip(bounds, bounds[1:]):
            if r <= l:
                continue
            cols = band[:, l:r]
            has = cols.any(0)
            if not has.any():
                continue
            top = np.argmax(cols[:, has], 0)
            bot = cols.shape[0] - np.argmax(cols[::-1, has], 0)
            # straight front and back: the median edge of the plot's columns (a rear line stays where it is)
            t = a if a > y0 else int(np.median(top))
            bt = b if b < y1 else int(np.median(bot))
            if bt - t < 2:
                continue
            n += 1
            out[t:bt, l:r] = n
    # back to the block's own frame
    Minv = cv2.invertAffineTransform(M)
    back = _rotate(out.astype(np.float32), Minv, (w, h), True).astype(np.int32)
    reach = ndi.distance_transform_edt(~block) * gsd <= FRONT_STRAIGHTEN_M
    back[~reach] = 0
    # block land outside every plot: a thin sliver beyond a straight front is road verge;
    # anything else (a deep notch) goes to the nearest plot
    miss = block & (back == 0)
    verge = np.zeros_like(block)
    if miss.any() and (back > 0).any():
        dist, (iy, ix) = ndi.distance_transform_edt(back == 0, return_indices=True)
        verge = miss & (dist * gsd <= FRONT_STRAIGHTEN_M) & ~ndi.binary_erosion(block, iterations=max(1, int(FRONT_STRAIGHTEN_M / gsd)))
        rest = miss & ~verge
        back[rest] = back[iy[rest], ix[rest]]
    elif miss.any():
        verge = miss
    return back, verge


def _merge_split_roofs(parcels, inst, max_roof_px):
    """A house must sit on one plot: plots sharing a house are merged."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for rid, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        ids = parcels[sl][inst[sl] == rid]
        ids = ids[ids > 0]
        if ids.size == 0 or ids.size > max_roof_px:
            continue
        cnt = np.bincount(ids)
        if cnt.max() / ids.size >= ROOF_KEEP_SHARE:
            continue
        owners = [i for i in np.nonzero(cnt)[0] if cnt[i] / ids.size >= 1 - ROOF_KEEP_SHARE]
        for o in owners[1:]:
            ra, rb = find(owners[0]), find(o)
            if ra != rb:
                parent[rb] = ra
    if not parent:
        return parcels
    lut = np.arange(parcels.max() + 1)
    for k in range(1, len(lut)):
        lut[k] = find(k)
    return lut[parcels]


def _rectangularity(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 1.0
    (_, _), (w, h), _ = cv2.minAreaRect(np.vstack(cs))
    return mask.sum() / max(w * h, 1)


def split_rectangular(block, gsd, depth=0):
    """Split an L / T / U shaped block at its deepest inner corner, straight along one of the
    block's grid directions, choosing the cut that leaves the most rectangular parts; repeat on
    the parts. Returns a label raster of parts (1..n) on the block crop."""
    if depth > 4 or _rectangularity(block) >= RECT_OK or block.sum() * gsd * gsd < 2 * MIN_PART_M2:
        return block.astype(np.int32)
    cs, _ = cv2.findContours(block.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c, returnPoints=False)
    try:
        defects = cv2.convexityDefects(c, hull)
    except cv2.error:
        defects = None
    if defects is None:
        return block.astype(np.int32)
    ang = np.deg2rad(_block_angle(block))
    dirs = [(np.cos(ang), np.sin(ang)), (-np.sin(ang), np.cos(ang))]
    best, best_score = None, _rectangularity(block)
    for s_, e_, f_, d_ in sorted(defects.reshape(-1, 4), key=lambda d: -d[3])[:4]:
        if d_ / 256.0 * gsd < MIN_DEFECT_M:
            continue
        px, py = c[f_][0]
        for dx, dy in dirs:
            L = max(block.shape) * 2
            line = np.zeros(block.shape, np.uint8)
            cv2.line(line, (int(px - dx * L), int(py - dy * L)), (int(px + dx * L), int(py + dy * L)), 1, 2)
            parts, n = ndi.label(block & ~line.astype(bool))
            if n < 2:
                continue
            sizes = np.bincount(parts.ravel())[1:] * gsd * gsd
            big = [i + 1 for i, a in enumerate(sizes) if a >= MIN_PART_M2]
            if len(big) < 2:
                continue
            score = np.average([_rectangularity(parts == i) for i in big], weights=[sizes[i - 1] for i in big])
            if score > best_score + 0.05:
                best, best_score = (parts, big), score
    if best is None:
        return block.astype(np.int32)
    parts, big = best
    # small crumbs and the cut line itself go to the nearest big part
    lab = np.zeros(block.shape, np.int32)
    for k, i in enumerate(big, start=1):
        lab[parts == i] = k
    miss = block & (lab == 0)
    if miss.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab[miss] = lab[iy[miss], ix[miss]]
    out = np.zeros(block.shape, np.int32)
    n = 0
    for k in range(1, len(big) + 1):
        sub = split_rectangular(lab == k, gsd, depth + 1)
        out[sub > 0] = sub[sub > 0] + n
        n += int(sub.max())
    return out


def _layout_plotted(block, roofs, edges, gsd, typ_global):
    """Plotted colony block: split into rectangular parts first, each laid out in its own direction."""
    parts = split_rectangular(block, gsd)
    if parts.max() <= 1:
        return _layout_block(block, roofs, edges, gsd, typ_global)
    plots = np.zeros(block.shape, np.int32)
    verge = np.zeros_like(block)
    n = 0
    for k in range(1, parts.max() + 1):
        part = parts == k
        p, v = _layout_block(part, np.where(part, roofs, 0), edges, gsd, typ_global)
        take = (p > 0) & (plots == 0) & (part | ~block)
        plots[take] = p[take] + n
        verge |= v
        n = int(plots.max())
    return plots, verge & ~(plots > 0)


def _organic_block(block, roofs, edges, gsd):
    """Unplanned block: each house grows its plot out to the visible walls (the organic method)."""
    from parcel_extract import parcel_raster          # late import: parcel_extract imports this module
    return parcel_raster(~block, roofs, edges, np.zeros(block.shape, np.uint8), gsd * gsd, gsd) * block


def layout_parcels(corridors, inst, edges, valid, gsd):
    """(parcel id raster, road mask). corridors: bool; inst: roof ids; edges: 0..1. The road mask
    comes back straightened: plot fronts are straight lines, so verge slivers join the road and
    small bumps of road into a plot front join the plot."""
    land = ~corridors & valid
    blocks, _ = ndi.label(land)
    # blocks too big to be one: add the streets the imagery hides, then relabel
    added = np.zeros_like(land)
    for b, sl in enumerate(ndi.find_objects(blocks), start=1):
        if sl is not None and (blocks[sl] == b).sum() * gsd * gsd >= BIG_BLOCK_M2:
            block = blocks[sl] == b
            r = np.where(block, inst[sl], 0)
            if (r > 0).sum() / block.sum() >= PLOTTED_MIN_COVER:      # only in a dense colony
                added[sl] |= hidden_streets(block, r, gsd)
    added &= ~ndi.binary_dilation(inst > 0, iterations=1)
    if added.any():
        corridors = corridors | added
        land = ~corridors & valid
        blocks, _ = ndi.label(land)
    parcels = np.zeros(land.shape, np.int32)
    kind_raster = np.zeros(land.shape, np.uint8)   # 1 plotted, 2 campus, 3 sparse, 4 organic
    road = corridors.copy()
    # the survey's typical plot width, for blocks with too few houses to measure their own
    roof_w = []
    for sl in ndi.find_objects(inst):
        if sl is not None:
            roof_w.append(min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * gsd)
    typ_global = float(np.clip(np.median(roof_w) + 2 * SIDE_MARGIN_M, *PLOT_WIDTH_RANGE_M)) if roof_w else PLOT_WIDTH_DEFAULT_M
    # a straightened plot front may take a little road edge, never the middle of a lane
    from skimage.morphology import skeletonize
    road_depth = ndi.distance_transform_edt(corridors) * gsd
    lane_core = cv2.dilate(skeletonize(corridors).astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=max(1, int(1.5 / gsd))) > 0
    may_take = corridors & (road_depth <= FRONT_TAKE_MAX_M) & ~lane_core
    next_id = 0
    kinds = {}
    for b, sl in enumerate(ndi.find_objects(blocks), start=1):
        if sl is None:
            continue
        block = blocks[sl] == b
        if block.sum() * gsd * gsd < MIN_BLOCK_M2:
            continue
        # pad the crop so a straightened plot front can step a little into the road
        pad = int(FRONT_STRAIGHTEN_M / gsd) + 2
        sl = (slice(max(0, sl[0].start - pad), sl[0].stop + pad), slice(max(0, sl[1].start - pad), sl[1].stop + pad))
        block = blocks[sl] == b
        roofs = np.where(block, inst[sl], 0)
        kind = block_type(block, roofs, gsd)
        kinds[kind] = kinds.get(kind, 0) + 1
        kind_raster[sl][block] = KIND_CODE[kind]
        if kind == "campus":
            plots, verge = block.astype(np.int32), np.zeros_like(block)      # one walled compound
        elif kind in ("organic", "sparse"):
            plots, verge = _organic_block(block, roofs, edges[sl], gsd), np.zeros_like(block)
        else:
            plots, verge = _layout_plotted(block, roofs, edges[sl], gsd, typ_global)
        plots = np.where(plots > 0, plots + next_id, 0)
        # a plot takes road pixels only where no other block's plot already sits
        take = (plots > 0) & (block | ((parcels[sl] == 0) & may_take[sl] & valid[sl]))
        parcels[sl][take] = plots[take]
        road[sl] |= verge
        road[sl] &= ~take
        next_id = max(next_id, int(plots.max()))
    parcels = _merge_split_roofs(parcels, inst, int(MERGE_MAX_ROOF_M2 / (gsd * gsd)))
    # specks below the minimum plot size (a corner cut off by a lane) go to whatever surrounds them
    sizes = np.bincount(parcels.ravel())
    specks = np.nonzero(sizes * gsd * gsd < MIN_PLOT_M2)[0]
    parcels[np.isin(parcels, specks[specks > 0])] = 0
    # compact ids
    ids = np.unique(parcels[parcels > 0])
    lut = np.zeros(parcels.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    layout_parcels.last_block_kinds = kinds
    layout_parcels.last_kind_raster = kind_raster
    return lut[parcels], road & ~(parcels > 0)
