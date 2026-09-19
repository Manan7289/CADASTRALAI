"""Fill in houses the roof model missed, using the land-cover model's building map.

The stacked roof model draws clean, separate outlines but was trained on one Indian
sector, so it skips some roofs (red tile, dark, apartment blocks). The land-cover model
learned buildings from 44 countries and finds most of them, but it only says "building
here", not where one house ends and the next begins. So: take land-cover building areas
the roof model left empty, drop thin edge slivers around roofs it already drew, split
what is left into houses (watershed on distance to the edge), and keep the pieces big
and compact enough to be a structure. Filled houses are tagged so the reviewer can
check them.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

MIN_FILL_M2 = 20          # smaller pieces are usually noise or a roof-model edge sliver
SLIVER_WIDTH_M = 2.0      # land-cover edges hugging an existing roof are removed by opening this wide
GAP_M = 0.6               # keep this much gap to roofs the model already drew
NECK_M = 1.0              # a lone blob is split only where it narrows by this much (half-width) between two humps
ROW_CONTEXT_M = 40        # a blob with roof-model houses within this distance...
ROW_CONTEXT_HOUSES = 3    # ...at least this many, is a row of touching houses: split into house widths
HOUSE_W_DEFAULT_M = 10
MIN_SOLIDITY = 0.75       # area / convex-hull area; ragged blobs are not structures (a hull, not the
                          # grid-aligned box, so buildings on diagonal streets are not thrown away)
MIN_WIDTH_M = 3.5         # a house is at least this wide; thinner pieces are gaps between roofs


def fill_missed_roofs(roofs, building_prob, valid, gsd, thresh=0.5):
    """roofs: (H,W) int instance ids from the roof model; building_prob: (H,W) 0..1.
    Returns (combined instance raster, boolean mask of the filled-in houses)."""
    roofs = np.asarray(roofs, np.int32)
    px_m2 = gsd * gsd
    cand = (building_prob >= thresh) & valid
    cand &= ~ndi.binary_dilation(roofs > 0, iterations=max(1, int(round(GAP_M / gsd))))
    r = max(1, int(round(SLIVER_WIDTH_M / gsd / 2)))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    cand = ndi.binary_opening(cand, structure=disk)
    cand = ndi.binary_fill_holes(cand)

    dist = ndi.distance_transform_edt(cand)
    comps, _ = ndi.label(cand)
    # Seeds. A blob among houses the roof model already found (a row of touching houses the land
    # cover sees as one) is split into house-width pieces, the width taken from those houses. A blob
    # with no such neighbours (a metro station, a mall, an apartment block) is one building unless
    # it narrows between two humps by NECK_M.
    from skimage.morphology import h_maxima
    humps, _ = ndi.label(h_maxima(dist, max(1.0, NECK_M / gsd)) & cand)
    widths = [min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in ndi.find_objects(roofs) if sl is not None]
    house_w = float(np.median(widths)) if len(widths) >= 5 else HOUSE_W_DEFAULT_M / gsd
    near = _count_near(roofs, int(ROW_CONTEXT_M / gsd))
    peaks = peak_local_max(dist, min_distance=max(2, int(house_w / 2)), threshold_abs=1.5 / gsd / 2,
                           labels=comps, exclude_border=False)
    row_seeds = np.zeros(cand.shape, np.int32)
    row_seeds[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    markers = np.zeros(cand.shape, np.int32)
    n = 0
    for c, sl in enumerate(ndi.find_objects(comps), start=1):
        if sl is None:
            continue
        m = comps[sl] == c
        src = row_seeds[sl] if near[sl][m].max() >= ROW_CONTEXT_HOUSES else humps[sl]
        ids = np.unique(src[m & (src > 0)])
        if ids.size == 0:
            ys, xs = np.nonzero(m)
            n += 1
            markers[sl[0].start + ys[len(ys) // 2], sl[1].start + xs[len(xs) // 2]] = n
            continue
        for v in ids:
            n += 1
            markers[sl][m & (src == v)] = n
    pieces = watershed(-dist, markers, mask=cand)

    out = roofs.copy()
    fill = np.zeros(roofs.shape, bool)
    nxt = int(roofs.max()) + 1
    for pid, sl in enumerate(ndi.find_objects(pieces), start=1):
        if sl is None:
            continue
        m = pieces[sl] == pid
        area = m.sum()
        if (area * px_m2 < MIN_FILL_M2 or area / _hull_area(m) < MIN_SOLIDITY
                or 2 * ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max() * gsd < MIN_WIDTH_M):
            continue
        out[sl][m] = nxt
        fill[sl][m] = True
        nxt += 1
    return out, fill


def _hull_area(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cv2.contourArea(cv2.convexHull(np.vstack(cs))), 1.0)


def _count_near(roofs, radius_px):
    """Per pixel: how many distinct roof-model houses have a pixel within radius_px (box window)."""
    count = np.zeros(roofs.shape, np.int32)
    for sl in ndi.find_objects(roofs):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - radius_px), sl[0].stop + radius_px
        x0, x1 = max(0, sl[1].start - radius_px), sl[1].stop + radius_px
        count[y0:y1, x0:x1] += 1
    return count
