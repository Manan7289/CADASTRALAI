"""Segmentation -> cadastral features: blocks, parcels, building footprints and
access corridors, as clean GIS polygons.

Input is the U-Net's per-pixel class probabilities on a georeferenced grid
(see segment.py), plus the RGB and, when the survey has one, the nDSM.

Why this produces topologically clean parcels by construction: parcels are
made as a *label raster* (every land pixel gets exactly one parcel id) and
only then vectorised, so neighbouring parcels share their boundary exactly --
no overlaps or gaps to digitise away. topology.py still validates the result
(simplification and the road clip can introduce problems) and repairs it.

Steps
  1. Access corridors: road/impervious pixels that form long connected strips
     (small isolated paved patches are courtyards, not roads). Width comes from
     the distance transform along the skeleton, so narrow lanes are flagged.
  2. Blocks: connected land between corridors.
  3. Building instances: building mask split into individual roofs by a
     watershed on the distance transform, sharpened by roof-edge strength --
     colour/texture edges from the image, and height steps from the nDSM when
     available (abutting roofs at different heights separate cleanly).
  4. Parcels: within each block, a marker-controlled watershed grows every
     roof over the surrounding open land up to the strongest visible
     boundary (walls, fences, colour/height edges). Every land pixel ends up
     in exactly one parcel.
  5. Vectorise (rasterio.features.shapes), coverage-preserving simplify,
     attributes, confidence.
"""
import numpy as np
import rasterio.features
import shapely
from affine import Affine
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping
from skimage.morphology import remove_small_objects, remove_small_holes, skeletonize
from skimage.segmentation import watershed
from skimage.filters import sobel

CLS = {"clutter": 0, "building": 1, "road": 2, "low_veg": 3, "tree": 4}

MIN_CORRIDOR_M2 = 60       # paved blobs smaller than this are yards/courtyards, not access corridors
MIN_CORRIDOR_ELONGATION = 3.0   # skeleton length / mean width
MIN_CORRIDOR_LENGTH_M = 10
MIN_BUILDING_M2 = 12
MIN_BLOCK_M2 = 30
NARROW_LANE_M = 3.0        # below this a corridor is a narrow access lane (PS: "narrow access roads")
ROOF_SEPARATION_M = 2.0    # min distance between two roof-instance seeds
SIMPLIFY_M = 0.35


def _px_area(transform):
    return abs(transform.a * transform.e)


def corridor_mask(labels, px_m2, gsd):
    """Keep paved regions that are strip-like: their skeleton is long relative
    to their typical width (a road cross or lane network qualifies at any size;
    a square courtyard does not)."""
    road = ndi.binary_opening(labels == CLS["road"], iterations=1)
    lab, n = ndi.label(road)
    keep = np.zeros(n + 1, dtype=bool)
    sizes = ndi.sum(road, lab, index=np.arange(n + 1)) * px_m2
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or sizes[i] < MIN_CORRIDOR_M2:
            continue
        region = lab[sl] == i
        skel_len_m = skeletonize(region).sum() * gsd
        mean_width_m = sizes[i] / max(skel_len_m, 1e-6)
        if skel_len_m >= MIN_CORRIDOR_LENGTH_M and skel_len_m >= MIN_CORRIDOR_ELONGATION * mean_width_m:
            keep[i] = True
    return keep[lab]


def edge_strength(rgb, ndsm=None, ndsm_weight=2.0):
    """0..1 boundary evidence: colour/texture edges, plus height steps if we
    have a surface model."""
    gray = rgb.astype(np.float32).mean(-1) / 255.0
    e = sobel(ndi.gaussian_filter(gray, 1.0))
    for c in range(3):
        e = np.maximum(e, sobel(ndi.gaussian_filter(rgb[..., c].astype(np.float32) / 255.0, 1.0)))
    e = e / (np.percentile(e, 99) + 1e-6)
    if ndsm is not None:
        h = sobel(ndi.gaussian_filter(ndsm.astype(np.float32), 1.0))
        h = h / (np.percentile(h, 99) + 1e-6)
        e = np.maximum(e, ndsm_weight * h)
    return np.clip(e, 0, 1)


def building_instances(labels, edges, gsd, px_m2):
    bmask = labels == CLS["building"]
    bmask = remove_small_holes(ndi.binary_opening(bmask, iterations=1), max_size=int(4 / px_m2))
    bmask = remove_small_objects(bmask, max_size=int(MIN_BUILDING_M2 / px_m2))
    dist = ndi.distance_transform_edt(bmask)
    # interior-ness discounted by roof edges, so seeds don't sit on a seam between two roofs
    score = dist * (1.0 - 0.8 * edges)
    size = max(3, int(ROOF_SEPARATION_M / gsd))
    peaks = (score == ndi.maximum_filter(score, size=size)) & (dist > 1.0 / gsd)
    markers, _ = ndi.label(ndi.binary_dilation(peaks, iterations=2) & bmask)
    inst = watershed(edges - dist / (dist.max() + 1e-6), markers, mask=bmask)
    # drop instances too small to be a structure (merge handled later by topology)
    sizes = np.bincount(inst.ravel())
    small = np.where(sizes * px_m2 < MIN_BUILDING_M2)[0]
    inst[np.isin(inst, small[small > 0])] = 0
    return inst


def parcel_raster(corridors, inst, edges, labels, px_m2):
    land = ~corridors
    blocks, n_blocks = ndi.label(land)
    parcels = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for b, sl in enumerate(ndi.find_objects(blocks), start=1):
        if sl is None:
            continue
        block = blocks[sl] == b
        if block.sum() * px_m2 < MIN_BLOCK_M2:
            continue
        seeds = np.where(block, inst[sl], 0)
        ids = np.unique(seeds[seeds > 0])
        if len(ids) == 0:
            # an unbuilt block is one open-land parcel -- no invented subdivision
            parcels[sl][block] = next_id
            next_id += 1
            continue
        relabel = np.zeros(seeds.max() + 1, dtype=np.int32)
        relabel[ids] = np.arange(next_id, next_id + len(ids))
        grown = watershed(edges[sl], relabel[seeds], mask=block)
        parcels[sl][block] = grown[block]
        next_id += len(ids)
    return parcels


def corridor_attributes(corridors, gsd):
    """Centreline skeleton + width -> list of corridor segments with width class."""
    skel = skeletonize(corridors)
    width = ndi.distance_transform_edt(corridors) * 2 * gsd
    w = width[skel]
    return {
        "centreline_px": int(skel.sum()),
        "length_m": round(float(skel.sum() * gsd), 1),
        "median_width_m": round(float(np.median(w)), 2) if w.size else None,
        "narrow_lane_length_m": round(float((w < NARROW_LANE_M).sum() * gsd), 1),
        "skeleton": skel, "width": width,
    }


def vectorise(label_raster, transform):
    """Polygonise a label raster (0 = nodata) and simplify all polygons as one
    coverage, so shared edges stay shared after simplification."""
    geoms = {}
    for geom, val in rasterio.features.shapes(label_raster.astype(np.int32), mask=label_raster > 0,
                                              transform=transform, connectivity=4):
        g = shape(geom)
        geoms[int(val)] = g if int(val) not in geoms else geoms[int(val)].union(g)
    ids = sorted(geoms)
    polys = shapely.coverage_simplify(np.array([geoms[i] for i in ids]), SIMPLIFY_M)
    return dict(zip(ids, polys))


def fill_unassigned(label_raster, valid):
    """Give every valid pixel without a label the label of its nearest labelled
    pixel (e.g. land fragments too small to be a block), so the output tiles
    the whole survey area with no holes."""
    missing = valid & (label_raster == 0)
    if not missing.any() or not (label_raster > 0).any():
        return label_raster
    _, (iy, ix) = ndi.distance_transform_edt(label_raster == 0, return_indices=True)
    out = label_raster.copy()
    out[missing] = label_raster[iy[missing], ix[missing]]
    return out


def merge_detached_pieces(label_raster):
    """Each label keeps only its largest connected piece; every other piece is
    relabelled to the neighbouring label it shares the longest border with, so
    no parcel ends up as a multipart polygon."""
    out = label_raster.copy()
    for lab_id, sl in enumerate(ndi.find_objects(out), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        region = out[sl] == lab_id
        comps, n = ndi.label(region)
        if n <= 1:
            continue
        sizes = np.bincount(comps.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        for c in range(1, n + 1):
            if c == keep:
                continue
            piece = comps == c
            ring = ndi.binary_dilation(piece) & ~piece
            neighbours = out[sl][ring]
            neighbours = neighbours[(neighbours != lab_id) & (neighbours > 0)]
            if neighbours.size:
                out[sl][piece] = np.bincount(neighbours).argmax()
    return out


def extract(probs, rgb, transform: Affine, ndsm=None, valid=None):
    """probs: (C,H,W) class probabilities; rgb: (H,W,3) uint8; transform maps
    pixel -> projected metres (UTM). Returns dict of feature lists (UTM
    geometries) plus summary stats."""
    gsd = abs(transform.a)
    px_m2 = _px_area(transform)
    labels = probs.argmax(0).astype(np.uint8)
    maxp = probs.max(0)

    edges = edge_strength(rgb, ndsm)
    corridors = corridor_mask(labels, px_m2, gsd)
    inst = building_instances(labels, edges, gsd, px_m2)
    parcels = parcel_raster(corridors, inst, edges, labels, px_m2)
    corr = corridor_attributes(corridors, gsd)

    valid = np.ones(labels.shape, bool) if valid is None else valid
    corridor_id = int(parcels.max()) + 1
    cover = np.where(corridors, corridor_id, parcels).astype(np.int32)
    # mask nodata (e.g. the warp border) BEFORE merging, since masking can split a parcel into pieces
    cover = merge_detached_pieces(np.where(valid, fill_unassigned(cover, valid), 0))
    parcels = np.where(cover == corridor_id, 0, cover)
    cover_polys = vectorise(cover, transform)
    corridor_polys = {1: cover_polys.pop(corridor_id)} if corridor_id in cover_polys else {}
    parcel_polys = cover_polys
    building_polys = vectorise(inst, transform)

    # per-parcel stats straight from the rasters
    idx = np.arange(parcels.max() + 1)
    n_px = np.bincount(parcels.ravel(), minlength=len(idx))
    built_px = np.bincount(parcels.ravel(), weights=(labels == CLS["building"]).ravel(), minlength=len(idx))
    veg_px = np.bincount(parcels.ravel(), weights=np.isin(labels, [CLS["low_veg"], CLS["tree"]]).ravel(), minlength=len(idx))
    conf_sum = np.bincount(parcels.ravel(), weights=maxp.ravel(), minlength=len(idx))
    boundary = parcels != ndi.grey_erosion(parcels, size=3)
    edge_on_boundary = np.bincount(parcels[boundary], weights=edges[boundary], minlength=len(idx))
    boundary_px = np.bincount(parcels[boundary], minlength=len(idx))
    touches_corr = ndi.binary_dilation(corridors, iterations=max(1, int(1.0 / gsd)))
    frontage = np.bincount(parcels[touches_corr & (parcels > 0)], minlength=len(idx)) > 0

    parcel_features = []
    for pid, geom in parcel_polys.items():
        if geom.is_empty:
            continue
        built = built_px[pid] / max(n_px[pid], 1)
        veg = veg_px[pid] / max(n_px[pid], 1)
        seg_conf = conf_sum[pid] / max(n_px[pid], 1)
        boundary_support = edge_on_boundary[pid] / max(boundary_px[pid], 1)
        confidence = round(float(0.6 * seg_conf + 0.4 * min(1.0, 2 * boundary_support)), 3)
        cover = "Built-up" if built >= 0.35 else ("Vegetated open land" if veg >= 0.5 else "Open / vacant land")
        parcel_features.append({
            "id": pid, "geometry": geom, "area_m2": round(geom.area, 1), "perimeter_m": round(geom.length, 1),
            "built_pct": round(float(100 * built), 1), "veg_pct": round(float(100 * veg), 1), "landcover": cover,
            "road_frontage": bool(frontage[pid]), "confidence": confidence,
        })

    building_features = [{"id": bid, "geometry": g, "area_m2": round(g.area, 1)}
                         for bid, g in building_polys.items() if not g.is_empty]
    corridor_features = [{"id": 1, "geometry": g,
                          "median_width_m": corr["median_width_m"], "length_m": corr["length_m"]}
                         for g in corridor_polys.values() if not g.is_empty]

    stats = {
        "gsd_m": gsd,
        "parcels": len(parcel_features), "buildings": len(building_features),
        "corridor_length_m": corr["length_m"], "corridor_median_width_m": corr["median_width_m"],
        "narrow_lane_length_m": corr["narrow_lane_length_m"],
        "class_fraction": {k: round(float((labels == v).mean()), 3) for k, v in CLS.items()},
        "used_height": ndsm is not None,
    }
    return {"parcels": parcel_features, "buildings": building_features, "corridors": corridor_features,
            "stats": stats, "rasters": {"labels": labels, "parcels": parcels, "corridors": corridors,
                                        "buildings": inst, "edges": edges, "skeleton": corr["skeleton"]}}
