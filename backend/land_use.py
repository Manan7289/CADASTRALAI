"""Suggested land use for each parcel (PS: "classification of land-use features", "mixed land use").

Land cover says what covers the ground (roof, road, tree); land use says what the plot is used
for. From imagery alone that can only be suggested, so each parcel gets a label AND the reason
for it, and the surveyor confirms it in review. The evidence, all from layers the AI already made:

    buildings    how many, how big the largest footprint is, what share of the plot they cover
    height       storeys from the surface model (DSM - DTM), when the survey has one
    frontage     how wide the road in front is (a plot on a 20 m road vs a 4 m lane)
    land cover   what the open part of the plot is (trees and grass, bare earth, paving, water)

Rules, first match wins:
    Water body                         water covers most of the plot
    Open space / green                 no building; mostly trees, grass or farmland
    Vacant plot                        no building; bare or paved ground
    Institutional / large complex      a footprint of 1500 m2+, or a very large plot with buildings
    Residential - apartments           a footprint of 300 m2+ and 4+ storeys (needs height)
    Commercial / mixed use             on a main road (12 m+), plot built over edge to edge
    Residential                        everything else with a house on it
"""
import numpy as np
from scipy import ndimage as ndi

STOREY_M = 3.0
MAIN_ROAD_M = 12.0
BUILT_OVER = 0.70           # building share of the plot above which it is "built edge to edge"
LARGE_FOOTPRINT_M2 = 1500
LARGE_PLOT_M2 = 3000
APARTMENT_FOOTPRINT_M2 = 300
APARTMENT_STOREYS = 4

LAND_USES = ["Residential", "Residential - apartments", "Commercial / mixed use", "Institutional / large complex",
             "Vacant plot", "Open space / green", "Water body"]


def parcel_evidence(parcels, inst, corridors, road_width, gsd, ndsm=None):
    """Per parcel id: building count, largest footprint (m2), building share, frontage road width (m),
    median roof height (m, or None)."""
    n = int(parcels.max()) + 1
    px = gsd * gsd
    roof = inst > 0
    cover = np.bincount(parcels.ravel(), weights=roof.ravel(), minlength=n) / np.maximum(np.bincount(parcels.ravel(), minlength=n), 1)
    count = np.zeros(n, int)
    largest = np.zeros(n)
    sizes = np.bincount(inst.ravel())
    for rid, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None or sizes[rid] * px < 20:
            continue
        ids = parcels[sl][inst[sl] == rid]
        ids = ids[ids > 0]
        if not ids.size:
            continue
        owner = int(np.bincount(ids).argmax())
        count[owner] += 1
        largest[owner] = max(largest[owner], sizes[rid] * px)
    # frontage: width of the road at its centreline nearest to the plot's road side
    frontage = np.zeros(n)
    if corridors.any():
        from skimage.morphology import skeletonize
        skel = skeletonize(corridors)
        if skel.any():
            _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
            w_near = road_width[iy, ix]
            edge = ndi.binary_dilation(corridors, iterations=max(1, int(1.0 / gsd))) & (parcels > 0)
            if edge.any():
                frontage = ndi.maximum(w_near, labels=np.where(edge, parcels, 0), index=np.arange(n))
                frontage = np.nan_to_num(np.asarray(frontage, float))
    height = None
    if ndsm is not None:
        h = np.where(roof & np.isfinite(ndsm), ndsm, 0.0)
        vals = ndi.median(h, labels=np.where(roof, parcels, 0), index=np.arange(n))
        height = np.asarray(vals, float)
    return {"count": count, "largest": largest, "cover": cover, "frontage": frontage, "height": height}


def classify(ev, pid, area_m2, lc_shares):
    """(land use, reason, extra attributes) for one parcel. lc_shares: 8-class land-cover shares (0..1) or {}."""
    n, big, cover, front = int(ev["count"][pid]), float(ev["largest"][pid]), float(ev["cover"][pid]), float(ev["frontage"][pid])
    h = None if ev["height"] is None else float(ev["height"][pid])
    storeys = int(round(h / STOREY_M)) if h and h > 1.5 else None
    extra = {"buildings": n, "building_cover_pct": round(100 * cover, 1), "frontage_road_m": round(front, 1)}
    if storeys:
        extra.update({"height_m": round(h, 1), "storeys": storeys})
    green = sum(lc_shares.get(k, 0) for k in ("tree", "grass / scrub", "agriculture"))
    if lc_shares.get("water", 0) >= 0.5:
        return "Water body", "water covers most of the plot", extra
    if n == 0:
        if green >= 0.5:
            return "Open space / green", f"no building; {round(100 * green)}% trees, grass or farmland", extra
        return "Vacant plot", "no building on the plot", extra
    if big >= LARGE_FOOTPRINT_M2 or (area_m2 >= LARGE_PLOT_M2 and n >= 1):
        return "Institutional / large complex", (f"a {round(big)} m² building" if big >= LARGE_FOOTPRINT_M2
                                                   else f"a {round(area_m2)} m² plot"), extra
    if storeys and storeys >= APARTMENT_STOREYS and big >= APARTMENT_FOOTPRINT_M2:
        return "Residential - apartments", f"{storeys} storeys, {round(big)} m² footprint", extra
    if front >= MAIN_ROAD_M and cover >= BUILT_OVER:
        return "Commercial / mixed use", f"on a {round(front)} m road, {round(100 * cover)}% built over", extra
    reason = f"{n} house{'s' if n > 1 else ''}, {round(100 * cover)}% built over"
    if storeys:
        reason += f", {storeys} storey{'s' if storeys > 1 else ''}"
    return "Residential", reason, extra
