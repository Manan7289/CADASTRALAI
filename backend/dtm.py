"""DTM from a DSM: a progressive morphological ground filter (Zhang et al.,
IEEE TGRS 2003), for surveys that deliver only a surface model.

Openings with growing windows strip off objects narrower than the window
(cars, then trees, then buildings); a cell is kept as ground only while its
height drop at each window stays within a slope-scaled threshold. Non-ground
cells are then filled from surrounding ground by nearest-ground + smoothing,
and nDSM = DSM - DTM is the height above ground the segmentation model uses.
"""
import numpy as np
from scipy import ndimage as ndi

MAX_WINDOW_M = 40.0    # larger than the biggest building footprint side expected in the AOI
INITIAL_WINDOW_M = 1.0
SLOPE = 0.15           # terrain slope (rise/run) tolerated between windows
DH0_M = 0.3            # initial height-difference threshold
DH_MAX_M = 2.5         # cap: steps above this are always objects, never terrain


def ground_mask(dsm, gsd_m):
    surface = np.where(np.isfinite(dsm), dsm, np.nanmax(dsm))
    ground = np.isfinite(dsm)
    window_m, prev_window_m = INITIAL_WINDOW_M, 0.0
    dh = DH0_M
    while window_m <= MAX_WINDOW_M:
        size = max(3, int(round(window_m / gsd_m)) | 1)
        opened = ndi.grey_opening(surface, size=(size, size))
        ground &= (surface - opened) <= dh
        dh = min(DH_MAX_M, SLOPE * (window_m - prev_window_m) + DH0_M)
        surface = opened
        prev_window_m, window_m = window_m, window_m * 2
    return ground


def dtm_from_dsm(dsm, gsd_m, smooth_m=3.0):
    ground = ground_mask(dsm, gsd_m)
    if not ground.any():
        return np.full_like(dsm, np.nanmin(dsm))
    _, (iy, ix) = ndi.distance_transform_edt(~ground, return_indices=True)
    dtm = dsm[iy, ix]
    dtm = ndi.uniform_filter(dtm, size=max(3, int(smooth_m / gsd_m)))
    return np.minimum(dtm, np.where(np.isfinite(dsm), dsm, dtm))


def ndsm_from_dsm(dsm, gsd_m):
    dtm = dtm_from_dsm(dsm, gsd_m)
    return np.clip(dsm - dtm, 0, None), dtm
