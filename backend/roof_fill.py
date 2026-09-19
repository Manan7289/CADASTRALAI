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
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

MIN_FILL_M2 = 20          # smaller pieces are usually noise or a roof-model edge sliver
SLIVER_WIDTH_M = 2.0      # land-cover edges hugging an existing roof are removed by opening this wide
GAP_M = 0.6               # keep this much gap to roofs the model already drew
HOUSE_SPACING_M = 5.0     # min distance between two house centres when splitting a block
MIN_SOLIDITY = 0.6        # area / bounding-box area; ragged blobs are not structures
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
    peaks = peak_local_max(dist, min_distance=max(2, int(HOUSE_SPACING_M / gsd / 2)),
                           threshold_abs=1.5 / gsd / 2, labels=comps, exclude_border=False)
    markers = np.zeros(cand.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    # components too small to have a peak still get one seed
    for c, sl in enumerate(ndi.find_objects(comps), start=1):
        if sl is not None and not markers[sl][comps[sl] == c].any():
            ys, xs = np.nonzero(comps[sl] == c)
            markers[sl[0].start + ys[len(ys) // 2], sl[1].start + xs[len(xs) // 2]] = markers.max() + 1
    pieces = watershed(-dist, markers, mask=cand)

    out = roofs.copy()
    fill = np.zeros(roofs.shape, bool)
    nxt = int(roofs.max()) + 1
    for pid, sl in enumerate(ndi.find_objects(pieces), start=1):
        if sl is None:
            continue
        m = pieces[sl] == pid
        area = m.sum()
        if (area * px_m2 < MIN_FILL_M2 or area / m.size < MIN_SOLIDITY
                or 2 * ndi.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max() * gsd < MIN_WIDTH_M):
            continue
        out[sl][m] = nxt
        fill[sl][m] = True
        nxt += 1
    return out, fill
