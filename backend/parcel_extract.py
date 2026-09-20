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
from skimage.filters.rank import majority
from skimage.morphology import disk, remove_small_objects, remove_small_holes, skeletonize
from skimage.segmentation import watershed
from skimage.filters import sobel

import regularise
from topology import clean_polygon
from road_network import WIDTH_CLASSES, bridge_lane_gaps, road_segments
from plot_layout import KIND_LABEL, layout_parcels, nearest_parcels
import land_use

CLS = {"clutter": 0, "building": 1, "road": 2, "low_veg": 3, "tree": 4}
# 8-class land cover (land cover v2, OpenEarthMap classes); index 0 is unused
LANDCOVER8 = {1: "bare land", 2: "grass / scrub", 3: "paved / developed", 4: "road", 5: "tree",
              6: "water", 7: "agriculture", 8: "building"}

MIN_CORRIDOR_M2 = 60       # paved blobs smaller than this are yards/courtyards, not access corridors
MIN_CORRIDOR_ELONGATION = 3.0   # skeleton length / mean width
MIN_CORRIDOR_LENGTH_M = 10
MIN_LANE_PIECE_M2 = 150     # a road/lane fragment cut off from the rest and smaller than this goes to its plot
PAVED_LOT_MIN_WIDTH_M = 10   # paved ground at least this wide all round is a lot / plaza, not a lane
MIN_BUILDING_M2 = 12
MIN_BLOCK_M2 = 30
NARROW_LANE_M = 3.0        # below this a corridor is a narrow access lane (PS: "narrow access roads")
ROOF_SEPARATION_M = 2.0    # min distance between two roof-instance seeds
PLOT_REACH_M = 6.0         # how far past its roof a building's plot may extend (yard / setback)
MIN_OPEN_PLOT_M2 = 40      # open-land scraps smaller than this join the neighbouring plot
OPEN_PLOT_MAX_M2 = 900     # open land larger than this is split along visible edges
OPEN_PLOT_SPACING_M = 22   # typical spacing between separate open plots
OPEN_PLOT_MIN_HALFWIDTH_M = 3
OPEN_PLOT_COMPACTNESS = 0.02  # keeps open plots near their seed on flat ground; edges still steer the cut
SMOOTH_RADIUS_M = 0.3      # majority-filter radius on the label raster
ABSORB_ENCLOSED_M2 = 60    # enclosed regions up to this size merge into the surrounding plot
SIMPLIFY_M = 2.0           # coverage simplification: straight survey-like edges, shared edges kept shared
BUILDING_SIMPLIFY_M = 0.5
RECTANGULARITY = 0.82      # roof area / min rotated rectangle area above which a roof is drawn as a rectangle
PLAUSIBLE_PLOT_M2 = 1500   # plots larger than this lose confidence (likely several plots merged)
IRREGULAR_THINNESS = 0.25  # 4*pi*A/P^2 below this counts as an irregular shape


def _px_area(transform):
    return abs(transform.a * transform.e)


def corridor_mask(labels, px_m2, gsd, paved=None):
    """Keep paved regions that are strip-like: their skeleton is long relative
    to their typical width (a road cross or lane network qualifies at any size;
    a square courtyard does not).

    paved: optional mask of paved ground that is not labelled road. Land cover often
    calls inner colony lanes "paved" rather than "road", so these join the network;
    paved parts wider than a lane (parking lots, plazas) are left out."""
    road = labels == CLS["road"]
    if paved is not None:
        both = road | paved
        r = max(1, int(round(PAVED_LOT_MIN_WIDTH_M / gsd / 2)))
        wide = ndi.binary_opening(both, structure=np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r)
        road = road | (paved & ~wide)
    road = ndi.binary_opening(road, iterations=1)
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


def _open_land_seeds(region, gsd):
    """Seeds for splitting large open land: evenly spaced points inside the
    region (skipping points squeezed against its edge). The compact watershed
    then grows a plot around each seed, with the cut between neighbouring
    plots following the strongest visible edge (wall, fence, path) nearby."""
    dist = ndi.distance_transform_edt(region)
    step = max(3, int(OPEN_PLOT_SPACING_M / gsd))
    min_d = min(OPEN_PLOT_MIN_HALFWIDTH_M / gsd, 0.5 * dist.max())
    seeds = np.zeros(region.shape, np.int32)
    n = 0
    for y in range(step // 2, region.shape[0], step):
        for x in range(step // 2, region.shape[1], step):
            if region[y, x] and dist[y, x] >= min_d:
                n += 1
                seeds[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = n
    if n == 0:
        cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        seeds[cy, cx] = 1
    return np.where(region, seeds, 0)


def parcel_raster(corridors, inst, edges, labels, px_m2, gsd, split_open=True):
    """Every land pixel gets exactly one parcel id.

    Roofs only claim land within PLOT_REACH_M of their footprint (their own
    yard / setback); land beyond that reach is open land. Large open land is
    split into separate plots along visible edges rather than kept as one
    catch-all parcel that wraps around its neighbours."""
    land = ~corridors
    blocks, _ = ndi.label(land)
    parcels = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    reach_px = max(1, int(PLOT_REACH_M / gsd))
    for b, sl in enumerate(ndi.find_objects(blocks), start=1):
        if sl is None:
            continue
        block = blocks[sl] == b
        if block.sum() * px_m2 < MIN_BLOCK_M2:
            continue
        seeds = np.where(block, inst[sl], 0)
        ids = np.unique(seeds[seeds > 0])
        block_out = np.zeros(block.shape, np.int32)

        if len(ids):
            relabel = np.zeros(seeds.max() + 1, dtype=np.int32)
            relabel[ids] = np.arange(next_id, next_id + len(ids))
            near = block & (ndi.distance_transform_edt(seeds == 0) <= reach_px)
            grown = watershed(edges[sl], relabel[seeds], mask=near)
            block_out[near] = grown[near]
            next_id += len(ids)

        open_land = block & (block_out == 0)
        if open_land.any():
            comps, n = ndi.label(open_land)
            for c in range(1, n + 1):
                region = comps == c
                area = region.sum() * px_m2
                if area < MIN_OPEN_PLOT_M2 and len(ids):
                    continue  # small scraps are absorbed by the neighbouring plot below
                if area <= OPEN_PLOT_MAX_M2 or not split_open:   # open ground (park, maidan) stays whole
                    block_out[region] = next_id
                    next_id += 1
                    continue
                osd = _open_land_seeds(region, gsd)
                grown = watershed(edges[sl], osd, mask=region, compactness=OPEN_PLOT_COMPACTNESS)
                block_out[region] = np.where(grown[region] > 0, grown[region] + next_id - 1, 0)
                next_id += int(osd.max())
        parcels[sl][block] = block_out[block]
    return parcels


def smooth_labels(label_raster, radius_px):
    """Majority filter over the label raster: removes the pixel staircase and
    texture-noise fringes along boundaries before vectorising."""
    if radius_px < 1 or label_raster.max() >= 65535:
        return label_raster
    lab16 = label_raster.astype(np.uint16)
    smoothed = majority(lab16, disk(radius_px), mask=label_raster > 0)
    return np.where(label_raster > 0, smoothed, 0).astype(label_raster.dtype)


def resolve_enclosures(cover, corridor_id, px_m2, max_rounds=6):
    """A parcel must not have a hole.

    - A small enclosed region (an open-land scrap, a courtyard mis-read as a
      lane) is absorbed into the parcel around it.
    - A substantial enclosed region (a real plot surrounded by one big plot)
      is kept, and the surrounding plot is instead split in two by a straight
      cut through the enclosed region -- so neither plot swallows the other.
    """
    out = cover.copy()
    for _ in range(max_rounds):
        changed = False
        next_id = int(out.max()) + 1
        for lab_id, sl in enumerate(ndi.find_objects(out), start=1):
            if sl is None or lab_id == corridor_id:
                continue
            sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
            view = out[sl]
            region = view == lab_id
            holes = ndi.binary_fill_holes(region) & ~region
            if not holes.any():
                continue
            hole_lab, n = ndi.label(holes)
            cut_rows = []
            for h in range(1, n + 1):
                hole = hole_lab == h
                if hole.sum() * px_m2 <= ABSORB_ENCLOSED_M2 or not (view[hole] > 0).any():
                    view[hole & (view > 0)] = lab_id
                    changed = True
                else:
                    cut_rows.append(int(np.nonzero(hole)[0].mean()))
            if cut_rows:
                region = view == lab_id
                cut = np.zeros_like(region)
                for r in cut_rows:
                    cut[r] = True
                pieces, m = ndi.label(region & ~cut)
                if m >= 2:
                    sizes = np.bincount(pieces.ravel())[1:]
                    keep = int(np.argmax(sizes)) + 1
                    for p in range(1, m + 1):
                        if p != keep:
                            view[pieces == p] = next_id
                            next_id += 1
                    # the one-pixel cut line joins whichever piece lies just above it
                    cut_px = region & cut
                    ys, xs = np.nonzero(cut_px)
                    above = view[np.maximum(ys - 1, 0), xs]
                    view[ys, xs] = np.where((above > 0) & (above != corridor_id), above, lab_id)
                    changed = True
        if not changed:
            break
    return out


def regularise_building(poly):
    """Buildings are drawn the way a surveyor would: nearly rectangular roofs
    become their minimum rotated rectangle, others are simplified with
    near-straight angles removed."""
    if poly.is_empty:
        return poly
    rect = poly.minimum_rotated_rectangle
    if rect.area > 0 and poly.area / rect.area >= RECTANGULARITY:
        return rect
    simple = poly.simplify(BUILDING_SIMPLIFY_M, preserve_topology=True)
    return simple if simple.is_valid and not simple.is_empty else poly


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


def _absorb_small_pieces(out, lab_id, sl, min_px):
    """Pieces of lab_id smaller than min_px go to the neighbour they share the most border with."""
    sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
    comps, n = ndi.label(out[sl] == lab_id)
    sizes = np.bincount(comps.ravel())
    for c in range(1, n + 1):
        if sizes[c] >= min_px:
            continue
        piece = comps == c
        ring = ndi.binary_dilation(piece) & ~piece
        nb = out[sl][ring]
        nb = nb[(nb != lab_id) & (nb > 0)]
        if nb.size:
            out[sl][piece] = np.bincount(nb).argmax()


def merge_detached_pieces(label_raster, keep_multipart=None, min_piece_px=0):
    """Each label keeps only its largest connected piece; every other piece is
    relabelled to the neighbouring label it shares the longest border with, so
    no parcel ends up as a multipart polygon.

    keep_multipart: a label that may stay in pieces (the road/lane network is
    broken up by tree canopy and parked cars, and its pieces are still roads);
    only its fragments too short to be a lane are handed to neighbours."""
    out = label_raster.copy()
    for lab_id, sl in enumerate(ndi.find_objects(out), start=1):
        if sl is None:
            continue
        if lab_id == keep_multipart:
            _absorb_small_pieces(out, lab_id, sl, min_px=min_piece_px)
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


def parcel_confidence(geom, seg_conf, boundary_support):
    """How much to trust a parcel, 0..1, with the reasons it was marked down.

    Pixel certainty and visible-boundary support are necessary but not
    sufficient: a huge, hole-ridden or very irregular parcel can be made of
    'certain' pixels and still be wrong, so plausibility of the shape itself
    multiplies the score."""
    reasons = []
    base = 0.6 * seg_conf + 0.4 * boundary_support
    if seg_conf < 0.6:
        reasons.append("uncertain segmentation")
    if boundary_support < 0.35:
        reasons.append("weak visible boundary")
    plaus = 1.0
    if geom.area > PLAUSIBLE_PLOT_M2:
        plaus *= max(0.35, (PLAUSIBLE_PLOT_M2 / geom.area) ** 0.5)
        reasons.append("very large for one plot")
    thin = 4 * np.pi * geom.area / geom.length ** 2 if geom.length else 0
    if thin < IRREGULAR_THINNESS:
        plaus *= max(0.5, thin / IRREGULAR_THINNESS)
        reasons.append("irregular shape")
    parts = getattr(geom, "geoms", [geom])
    if any(len(p.interiors) for p in parts):
        plaus *= 0.5
        reasons.append("encloses another area")
    return round(float(base * plaus), 3), reasons


def instances_from_raster(inst, px_m2):
    """Roof instances supplied by a separate roof model (e.g. the stacked Mask R-CNN):
    relabel 1..n and drop fragments too small to be a structure."""
    inst = np.asarray(inst, dtype=np.int32).copy()
    sizes = np.bincount(inst.ravel())
    small = np.where(sizes * px_m2 < MIN_BUILDING_M2)[0]
    inst[np.isin(inst, small[small > 0])] = 0
    ids = np.unique(inst[inst > 0])
    remap = np.zeros(int(inst.max()) + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    return remap[inst]


def _parcel_landcover_label(built, shares):
    """Dominant use of a parcel from the 8-class land cover (built-up first)."""
    if built >= 0.35:
        return "Built-up"
    open_shares = {k: v for k, v in shares.items() if k != "building"}
    if not open_shares or max(open_shares.values()) < 0.2:
        return "Open / vacant land"
    top = max(open_shares, key=open_shares.get)
    return {"bare land": "Barren land", "grass / scrub": "Vegetated open land", "tree": "Vegetated open land",
            "agriculture": "Vegetated open land", "paved / developed": "Paved / developed open land",
            "road": "Paved / developed open land", "water": "Water"}.get(top, "Open / vacant land")


ENCROACH_MIN_M2 = 4.0       # a building part on the road smaller than this is outline noise
ENCROACH_MIN_DEPTH_M = 1.0  # ...and it must stick out at least this far (0.3 m imagery: 3+ pixels)
ENCROACH_MAX_SHARE = 0.35   # above this share of the building, the road and building layers disagree (detection issue)


def building_checks(building_polys, road_poly, inst, parcels, ndsm=None):
    """Per building: the plot it stands on, height/storeys (with a surface model), and whether it
    extends onto the road -- a small part on the road is a possible encroachment (PS: encroachments,
    narrow access roads); a large part means the road and building detections disagree."""
    out = {}
    for bid, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None or bid not in building_polys:
            continue
        m = inst[sl] == bid
        ids = parcels[sl][m]
        ids = ids[ids > 0]
        c = {"parcel": int(np.bincount(ids).argmax()) if ids.size else 0}
        if ndsm is not None:
            h = ndsm[sl][m]
            h = h[np.isfinite(h)]
            if h.size and np.median(h) > 1.5:
                c["height_m"] = round(float(np.median(h)), 1)
                c["storeys"] = max(1, int(round(float(np.median(h)) / 3.0)))
        g = building_polys[bid]
        if road_poly is not None and not g.is_empty and g.intersects(road_poly):
            part = g.intersection(road_poly)
            ov = part.area
            if ov >= ENCROACH_MIN_M2:
                if ov / max(g.area, 1e-6) > ENCROACH_MAX_SHARE:
                    c["road_conflict"] = True
                else:
                    # how far it sticks out: the thin side of the part on the road
                    pieces = [p for p in getattr(part, "geoms", [part]) if p.area > 0]
                    big = max(pieces, key=lambda p: p.area)
                    r = big.minimum_rotated_rectangle
                    xs, ys = r.exterior.coords.xy
                    sides = sorted(((xs[i + 1] - xs[i]) ** 2 + (ys[i + 1] - ys[i]) ** 2) ** 0.5 for i in range(2))
                    if sides[0] >= ENCROACH_MIN_DEPTH_M:
                        c["encroachment_m2"] = round(ov, 1)
                        c["encroachment_depth_m"] = round(sides[0], 1)
        out[bid] = c
    return out


def extract(probs, rgb, transform: Affine, ndsm=None, valid=None, inst_override=None, landcover=None, inst_fill=None,
            paved=None, building_source="roof model", parcel_method="nearest", boundary=None):
    """probs: (C,H,W) class probabilities; rgb: (H,W,3) uint8; transform maps
    pixel -> projected metres (UTM). Returns dict of feature lists (UTM
    geometries) plus summary stats.

    inst_override: optional (H,W) roof-instance raster from a dedicated roof model;
    used instead of splitting the segmentation's building class.
    landcover: optional (H,W) 8-class land-cover raster (LANDCOVER8); adds a
    per-parcel land-cover breakdown and finer parcel land-use labels.
    inst_fill: optional (H,W) bool mask of houses in inst_override that were filled in
    from the land-cover building map (roof_fill.py); those footprints are tagged for review.
    paved: optional (H,W) bool mask of paved non-road ground; lane-shaped parts join the corridors."""
    gsd = abs(transform.a)
    px_m2 = _px_area(transform)
    labels = probs.argmax(0).astype(np.uint8)
    maxp = probs.max(0)

    edges = edge_strength(rgb, ndsm)
    corridors = corridor_mask(labels, px_m2, gsd, paved)
    inst = building_instances(labels, edges, gsd, px_m2) if inst_override is None else instances_from_raster(inst_override, px_m2)
    if parcel_method in ("layout", "nearest"):
        # blocks are only right if the lane network is closed: bridge gaps under trees / vehicles
        corridors, _ = bridge_lane_gaps(corridors, ndi.binary_dilation(inst > 0, iterations=2), gsd)

    # only nodata connected to the image edge is outside the survey; black pixels inside it (deep shadow) are not
    valid = np.ones(labels.shape, bool) if valid is None else ndi.binary_fill_holes(valid)
    if parcel_method == "nearest":
        # every piece of land to the nearest house in its block; roads separate (plot_layout.py)
        parcels, corridors = nearest_parcels(corridors, inst, valid, gsd, boundary)
    elif parcel_method == "layout":
        # blocks from the road network -> rows -> one plot per house, cut along wall lines (plot_layout.py)
        parcels, corridors = layout_parcels(corridors, inst, edges, valid, gsd)
    else:
        parcels = parcel_raster(corridors, inst, edges, labels, px_m2, gsd)
    corridor_id = int(parcels.max()) + 1
    cover = np.where(corridors, corridor_id, parcels).astype(np.int32)
    # mask nodata (e.g. the warp border) BEFORE merging, since masking can split a parcel into pieces
    cover = np.where(valid, fill_unassigned(cover, valid), 0)
    cover = smooth_labels(cover, int(round(SMOOTH_RADIUS_M / gsd)))
    # detached-piece merging can create new enclosures and vice versa, so settle both
    for _ in range(3):
        before = cover
        lane_px = int(MIN_LANE_PIECE_M2 / px_m2)
        cover = merge_detached_pieces(resolve_enclosures(merge_detached_pieces(cover, corridor_id, lane_px), corridor_id, px_m2),
                                      corridor_id, lane_px)
        if np.array_equal(before, cover):
            break
    parcels = np.where(cover == corridor_id, 0, cover)
    corridors = cover == corridor_id
    corr = corridor_attributes(corridors, gsd)
    cover_polys = vectorise(cover, transform)
    corridor_polys = {1: cover_polys.pop(corridor_id)} if corridor_id in cover_polys else {}
    # coverage simplification can pinch a plot into a figure-8; keep it one valid polygon
    parcel_polys = {k: (g if g.is_valid else clean_polygon(g)) for k, g in cover_polys.items()}
    building_polys = {k: regularise_building(g) for k, g in vectorise(inst, transform).items()}
    # Plot lines the way a surveyor draws them: straight, shared with the neighbour, along the
    # grid the houses sit on (see regularise.py). Sizes are left as found.
    parcel_polys = {k: (g if g.is_valid else clean_polygon(g))
                    for k, g in regularise.regularise(parcel_polys, building_polys, exclude=corridor_polys.get(1)).items()}
    road_poly = shapely.union_all(list(corridor_polys.values())) if corridor_polys else None
    bchecks = building_checks(building_polys, road_poly, inst, parcels, ndsm)

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

    kind_of = None
    if parcel_method == "layout" and getattr(layout_parcels, "last_kind_raster", None) is not None:
        kr = layout_parcels.last_kind_raster
        kc = np.zeros((len(idx), 5))
        np.add.at(kc, (parcels.ravel(), kr.ravel()), 1)
        kind_of = kc[:, 1:].argmax(1) + 1
    evidence = land_use.parcel_evidence(parcels, inst, corridors, corr["width"], gsd, ndsm)
    lc_counts = None
    if landcover is not None:
        lc = np.asarray(landcover, dtype=np.int64)
        lc_counts = np.zeros((len(idx), 9))
        np.add.at(lc_counts, (parcels.ravel(), np.clip(lc.ravel(), 0, 8)), 1)

    parcel_features = []
    for pid, geom in parcel_polys.items():
        if geom.is_empty:
            continue
        built = built_px[pid] / max(n_px[pid], 1)
        veg = veg_px[pid] / max(n_px[pid], 1)
        seg_conf = conf_sum[pid] / max(n_px[pid], 1)
        boundary_support = min(1.0, 2 * edge_on_boundary[pid] / max(boundary_px[pid], 1))
        confidence, reasons = parcel_confidence(geom, seg_conf, boundary_support)
        cover = "Built-up" if built >= 0.35 else ("Vegetated open land" if veg >= 0.5 else "Open / vacant land")
        feat = {
            "id": pid, "geometry": geom, "area_m2": round(geom.area, 1), "perimeter_m": round(geom.length, 1),
            "built_pct": round(float(100 * built), 1), "veg_pct": round(float(100 * veg), 1), "landcover": cover,
            "road_frontage": bool(frontage[pid]), "confidence": confidence, "confidence_notes": reasons,
        }
        if kind_of is not None:
            feat["layout"] = KIND_LABEL[int(kind_of[pid])]
        lc_sh = {}
        if lc_counts is not None:
            tot = max(lc_counts[pid, 1:].sum(), 1)
            lc_sh = {LANDCOVER8[k]: lc_counts[pid, k] / tot for k in LANDCOVER8}
        use, why, extra = land_use.classify(evidence, pid, geom.area, lc_sh)
        feat.update({"land_use": use, "land_use_reason": why, **extra})
        enc = [c for c in bchecks.values() if c["parcel"] == pid and c.get("encroachment_m2")]
        if enc:
            feat["encroachment_m2"] = round(sum(c["encroachment_m2"] for c in enc), 1)
            feat["confidence_notes"] = feat["confidence_notes"] + [f"building extends {feat['encroachment_m2']} m² onto the road (possible encroachment)"]
        if lc_counts is not None:
            tot = max(lc_counts[pid, 1:].sum(), 1)
            shares = {LANDCOVER8[k]: lc_counts[pid, k] / tot for k in LANDCOVER8}
            feat["land_cover_pct"] = {k: round(100 * v, 1) for k, v in shares.items() if v >= 0.01}
            feat["landcover"] = _parcel_landcover_label(built, shares)
            feat["veg_pct"] = round(100 * (shares["tree"] + shares["grass / scrub"] + shares["agriculture"]), 1)
        parcel_features.append(feat)

    filled = set()
    if inst_fill is not None:
        filled = set(np.unique(inst[np.asarray(inst_fill, bool) & (inst > 0)]).tolist())
    building_features = [{"id": bid, "geometry": g, "area_m2": round(g.area, 1),
                          "source": "land cover fill-in (check)" if bid in filled else building_source,
                          **{k: v for k, v in bchecks.get(bid, {}).items() if k != "parcel"}}
                         for bid, g in building_polys.items() if not g.is_empty]
    corridor_features = [{"id": 1, "geometry": g,
                          "median_width_m": corr["median_width_m"], "length_m": corr["length_m"]}
                         for g in corridor_polys.values() if not g.is_empty]
    road_lines = road_segments(corr["skeleton"], corr["width"], transform, gsd)

    stats = {
        "gsd_m": gsd,
        "parcels": len(parcel_features), "buildings": len(building_features),
        "buildings_filled": len([b for b in building_features if b["id"] in filled]),
        "encroachments": sum(1 for b in building_features if b.get("encroachment_m2")),
        "road_building_conflicts": sum(1 for b in building_features if b.get("road_conflict")),
        "corridor_length_m": corr["length_m"], "corridor_median_width_m": corr["median_width_m"],
        "narrow_lane_length_m": corr["narrow_lane_length_m"],
        "class_fraction": {k: round(float((labels == v).mean()), 3) for k, v in CLS.items()},
        "used_height": ndsm is not None,
    }
    stats["land_use_count"] = {u: sum(1 for f in parcel_features if f.get("land_use") == u) for u in land_use.LAND_USES}
    if landcover is not None:
        lcv = np.asarray(landcover)[valid]
        stats["land_cover_fraction"] = {name: round(float((lcv == k).mean()), 3) for k, name in LANDCOVER8.items()}
    stats["road_length_by_class_m"] = {c: round(sum(r["length_m"] for r in road_lines if r["class"] == c), 1)
                                       for _, c in WIDTH_CLASSES}
    return {"parcels": parcel_features, "buildings": building_features, "corridors": corridor_features,
            "roads": road_lines, "stats": stats, "rasters": {"labels": labels, "parcels": parcels, "corridors": corridors,
                                        "buildings": inst, "edges": edges, "skeleton": corr["skeleton"]}}
