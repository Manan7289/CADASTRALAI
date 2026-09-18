"""
Visible Vegetation & Barren Land Identification Module
======================================================
Implements high-accuracy optical (RGB) remote sensing indices:
1. Excess Green Index (ExG): ExG = 2g - r - b
2. Visible Atmospheric Resistant Index (VARI): VARI = (G - R) / (G + R - B + eps)
3. Green Leaf Index (GLI): GLI = (2G - R - B) / (2G + R + B + eps)
4. Soil Tone Index (STI) & Redness Index for Barren / Fallow Land:
   STI = (R - B) / (R + B + eps)
5. Canopy Texture Entropy / Roughness:
   Separates ruffled high-canopy trees/orchards from planar cropland/grass.
6. Agricultural Bund & Ridge Extraction:
   Demarcates field borders between adjacent farm / barren parcels.
"""
import math
from typing import Dict, List, Tuple, Any, Optional, Callable

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, mapping, Point, shape
from shapely.ops import unary_union


def compute_visible_vegetation_and_soil_indices(
    img_bgr: np.ndarray,
    building_mask: Optional[np.ndarray] = None,
    road_mask: Optional[np.ndarray] = None,
    meters_per_px: float = 0.3,
) -> Dict[str, np.ndarray]:
    """
    Compute optical vegetation, tree canopy, and barren soil masks from a standard RGB/BGR image.
    
    Parameters
    ----------
    img_bgr : np.ndarray
        Source BGR drone/aerial imagery.
    building_mask : Optional[np.ndarray]
        Binary mask of detected building footprints (dilated to prevent roof edge false-positives).
    road_mask : Optional[np.ndarray]
        Binary mask of road surfaces/corridors (dilated to prevent median/curb false-positives).
    meters_per_px : float
        Spatial resolution in meters per pixel.
        
    Returns
    -------
    dict containing:
      - 'exg': float32 Excess Green array
      - 'vari': float32 Visible Atmospheric Resistant Index
      - 'sti': float32 Soil Tone Index
      - 'veg_mask': uint8 (0 or 255) total active vegetation
      - 'tree_mask': uint8 (0 or 255) real tree canopy crowns (compact, elevated, shadow-paired)
      - 'crop_mask': uint8 (0 or 255) agricultural cropland / grassland
      - 'barren_mask': uint8 (0 or 255) barren land / bare soil / fallow earth
    """
    h, w = img_bgr.shape[:2]
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    tot = r + g + b + 1e-6

    # Normalized chromatic coordinates
    rn = r / tot
    gn = g / tot
    bn = b / tot

    # 1. Excess Green Index (ExG)
    exg = 2.0 * gn - rn - bn

    # 2. Visible Atmospheric Resistant Index (VARI)
    vari_denom = g + r - b
    vari_denom = np.where(np.abs(vari_denom) < 1e-4, 1e-4, vari_denom)
    vari = (g - r) / vari_denom

    # 3. Soil Tone Index (STI) for Barren Land & Bare Soil
    # Bare mineral soil, dry fallow earth, and clay/sand have strong red reflection over blue
    sti = (r - b) / (r + b + 1e-6)

    # 4. Color space transforms for refined discrimination
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # OpenCV: [0, 180], 0=Red, 30=Yellow, 60=Green
    sat = hsv[:, :, 1]  # [0, 255]
    val = hsv[:, :, 2]  # [0, 255]

    # Dilated building and road exclusion zones
    if building_mask is not None and np.count_nonzero(building_mask) > 0:
        b_dilated = cv2.dilate(building_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    else:
        b_dilated = np.zeros((h, w), dtype=np.uint8)

    if road_mask is not None and np.count_nonzero(road_mask) > 0:
        road_dilated = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    else:
        road_dilated = np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k_size = min(7, max(3, min(h, w) // 3))
    if k_size % 2 == 0:
        k_size += 1
    local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (k_size, k_size))
    local_sq = cv2.boxFilter(gray.astype(np.float32) ** 2, -1, (k_size, k_size))
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean ** 2))

    # Multi-scale canopy roughness (15x15 kernel to capture true canopy crown texture, not pixel noise)
    k15 = min(15, max(3, min(h, w) // 3))
    if k15 % 2 == 0:
        k15 += 1
    mean15 = cv2.boxFilter(gray.astype(np.float32), -1, (k15, k15))
    sq15 = cv2.boxFilter(gray.astype(np.float32) ** 2, -1, (k15, k15))
    std15 = np.sqrt(np.maximum(0.0, sq15 - mean15 ** 2))

    # ── Active Vegetation Mask ───────────────────────────────────────────────
    # True vegetation has positive ExG, green dominance over R & B, and green hue (26-90)
    is_green_hue = (hue >= 26) & (hue <= 90)
    veg_raw = (
        (exg > 0.030) &
        (g > r * 0.96) &
        (g > b * 1.01) &
        (is_green_hue | (vari > 0.02)) &
        (sat > 20) &
        (b_dilated == 0) &
        (road_dilated == 0)
    ).astype(np.uint8) * 255

    # Clean isolated noise
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    veg_mask = cv2.morphologyEx(veg_raw, cv2.MORPH_OPEN, kc)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # ── Tree Canopy vs Cropland vs Lawn Separation ──────────────────────────
    # 1. Shadow mapping & Canopy-Shadow Pairing:
    # Elevated tree crowns cast distinct cast-shadows (V < 52) on their adjacent ground.
    shadow_raw = ((val < 52) & (b_dilated == 0)).astype(np.uint8) * 255
    shadow_clean = cv2.morphologyEx(shadow_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # Kernel searching in the shadow cast direction and radial proximity
    k_shadow_size = min(21, max(5, min(h, w) // 4))
    if k_shadow_size % 2 == 0:
        k_shadow_size += 1
    kernel_shadow = np.zeros((k_shadow_size, k_shadow_size), dtype=np.uint8)
    center_s = k_shadow_size // 2
    for dy in range(k_shadow_size):
        for dx in range(k_shadow_size):
            if (dy >= center_s and dx >= center_s) or ((dy - center_s)**2 + (dx - center_s)**2 <= (center_s * 0.8)**2):
                kernel_shadow[dy, dx] = 1
    shadow_paired = cv2.dilate(shadow_clean, kernel_shadow)

    # 2. Lawn / Median suppression:
    # Turf grass in flat lawns and medians has high surface luminance (val > 105),
    # low-to-moderate texture (std15 < 15.0), and NO associated shadow pairing.
    is_flat_lawn = (veg_mask > 0) & (shadow_paired == 0) & (val > 105) & (std15 < 15.0)

    # 3. True tree canopy foliage:
    # Green vegetation that is NOT flat lawn, and exhibits shadow pairing,
    # or high multi-scale canopy texture (std15 > 15.0 with val < 115), or dark foliage (val < 90).
    tree_cond = (
        (veg_mask > 0) &
        (~is_flat_lawn) &
        (b_dilated == 0) &
        (road_dilated == 0) &
        ((shadow_paired > 0) | (std15 > 15.0) | (val < 90))
    )
    tree_raw = tree_cond.astype(np.uint8) * 255
    tree_mask = cv2.morphologyEx(tree_raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    crop_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(tree_mask))

    # ── Barren Land / Bare Soil / Fallow Earth Mask ──────────────────────────
    # Barren land has:
    # 1. Negative or near-zero ExG (absence of chlorophyll)
    # 2. Warm earth hue (H in [5, 26], orange/tan/brown/terracotta)
    # 3. High red-to-blue ratio (STI > 0.08) and R >= G
    # 4. Moderate saturation (distinguishes from neutral grey roads/asphalt)
    # 5. Low-to-moderate texture roughness (not building roofs with steep edges)
    # 6. Strictly excluded from buildings and paved roads
    # Soil hue range expanded: [5,30] captures terracotta, sandy-brown, ochre, and dry clay.
    is_soil_hue = (hue >= 5) & (hue <= 30)
    # Also catch yellowish-tan fallow land (hue 1-4 wraps around near red)
    is_warm_earth = (hue <= 4) | is_soil_hue
    barren_cond = (
        (veg_mask == 0) &
        (b_dilated == 0) &
        (road_dilated == 0) &
        (r > g * 0.97) &
        (r > b * 1.08) &
        (sti > 0.06) &
        (is_warm_earth | (sti > 0.14)) &
        (sat >= 15) & (sat <= 185) &
        (val >= 45) & (val <= 240) &
        (local_std < 36.0)
    )
    barren_raw = barren_cond.astype(np.uint8) * 255
    barren_mask = cv2.morphologyEx(barren_raw, cv2.MORPH_OPEN, kc)
    # Larger closing kernel merges adjacent small barren patches into continuous zones
    barren_mask = cv2.morphologyEx(barren_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    return {
        "exg": exg,
        "vari": vari,
        "sti": sti,
        "veg_mask": veg_mask,
        "tree_mask": tree_mask,
        "crop_mask": crop_mask,
        "barren_mask": barren_mask,
        "std15": std15,
    }


def detect_agricultural_bunds(
    img_bgr: np.ndarray,
    crop_mask: np.ndarray,
    barren_mask: np.ndarray,
    meters_per_px: float = 0.3
) -> List[np.ndarray]:
    """
    Detect agricultural field bunds (earthen ridges, furrows, and plot dividers)
    that demarcate farm and barren land parcel boundaries.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    open_land_mask = cv2.bitwise_or(crop_mask, barren_mask)

    # Enhance linear ridge features using morphological black-hat
    k_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    k_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_line_h)
    blackhat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_line_v)
    ridge_mag = cv2.addWeighted(blackhat_h, 0.5, blackhat_v, 0.5, 0)

    # Edge detection restricted to open land areas
    edges = cv2.Canny(ridge_mag, 25, 75)
    edges = cv2.bitwise_and(edges, open_land_mask)

    # Probabilistic Hough transform to find linear bunds
    min_len = int(15.0 / max(0.05, meters_per_px))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_len, maxLineGap=12)
    bund_segments = []
    if lines is not None:
        for ln in lines:
            pts = np.array(ln).ravel()
            if len(pts) >= 4:
                bund_segments.append(np.array([[int(pts[0]), int(pts[1])], [int(pts[2]), int(pts[3])]], dtype=np.int32))

    return bund_segments


def analyze_parcel_landcover(
    poly_px: Polygon,
    veg_mask: np.ndarray,
    tree_mask: np.ndarray,
    barren_mask: np.ndarray,
    bldg_area_px: float = 0.0,
    meters_per_px: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate high-resolution land cover proportions and classify parcel landuse.
    
    Returns
    -------
    dict with:
      - 'vegetation_cover_pct': float (0-100)
      - 'tree_cover_pct': float (0-100)
      - 'crop_cover_pct': float (0-100)
      - 'barren_cover_pct': float (0-100)
      - 'built_up_pct': float (0-100)
      - 'crop_canopy_index': float (0.0 to 1.0)
      - 'cultivable_area_m2': float
      - 'cultivable_area_acres': float
      - 'barren_area_m2': float
      - 'landuse': string classification
    """
    h, w = veg_mask.shape[:2]
    if poly_px.is_empty or poly_px.area < 1:
        return {
            "vegetation_cover_pct": 0.0,
            "tree_cover_pct": 0.0,
            "crop_cover_pct": 0.0,
            "barren_cover_pct": 0.0,
            "built_up_pct": 0.0,
            "crop_canopy_index": 0.0,
            "cultivable_area_m2": 0.0,
            "cultivable_area_acres": 0.0,
            "barren_area_m2": 0.0,
            "landuse": "Vacant / Open Land",
        }

    # Create polygon mask
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    coords = np.array(poly_px.exterior.coords, dtype=np.int32)
    cv2.fillPoly(poly_mask, [coords], 255)

    parcel_px_count = max(1, int(poly_px.area))
    m2_per_px2 = float(meters_per_px) ** 2

    # Sample inside parcel
    in_veg = cv2.bitwise_and(veg_mask, poly_mask)
    in_tree = cv2.bitwise_and(tree_mask, poly_mask)
    in_barren = cv2.bitwise_and(barren_mask, poly_mask)

    veg_count = int(np.count_nonzero(in_veg))
    tree_count = int(np.count_nonzero(in_tree))
    crop_count = max(0, veg_count - tree_count)
    barren_count = int(np.count_nonzero(in_barren))

    veg_pct = round((veg_count / parcel_px_count) * 100.0, 1)
    tree_pct = round((tree_count / parcel_px_count) * 100.0, 1)
    crop_pct = round((crop_count / parcel_px_count) * 100.0, 1)
    barren_pct = round((barren_count / parcel_px_count) * 100.0, 1)
    built_pct = round((bldg_area_px / parcel_px_count) * 100.0, 1)

    cultivable_m2 = round(crop_count * m2_per_px2, 1)
    cultivable_acres = round(cultivable_m2 / 4046.86, 4)
    barren_m2 = round(barren_count * m2_per_px2, 1)
    cci = round(veg_count / parcel_px_count, 3)

    # ── Land Use Classification Decision Logic ──────────────────────────────
    if built_pct >= 12.0:
        if bldg_area_px * m2_per_px2 >= 350.0:
            landuse = "Commercial / Retail Complex"
        elif veg_pct >= 25.0:
            landuse = "Residential Homestead / Farmhouse"
        else:
            landuse = "Residential / Built-up"
    elif tree_pct >= 30.0:
        landuse = "Tree Canopy / Orchard / Agro-Forestry"
    elif crop_pct >= 30.0 or (veg_pct >= 35.0):
        landuse = "Agricultural / Cultivated Cropland"
    elif barren_pct >= 30.0 or (barren_pct > veg_pct and barren_pct > 20.0):
        landuse = "Barren Land / Fallow Rural Ground"
    elif veg_pct >= 15.0:
        landuse = "Agricultural / Cultivated Cropland"
    else:
        landuse = "Vacant Plot / Open Land"

    return {
        "vegetation_cover_pct": veg_pct,
        "tree_cover_pct": tree_pct,
        "crop_cover_pct": crop_pct,
        "barren_cover_pct": barren_pct,
        "built_up_pct": built_pct,
        "crop_canopy_index": cci,
        "cultivable_area_m2": cultivable_m2,
        "cultivable_area_acres": cultivable_acres,
        "barren_area_m2": barren_m2,
        "landuse": landuse,
    }


def delineate_open_rural_parcels(
    crop_mask: np.ndarray,
    barren_mask: np.ndarray,
    bund_segments: List[np.ndarray],
    occupied_polys: List[Polygon],
    roi_box: Polygon,
    meters_per_px: float = 0.3,
    min_area_m2: float = 120.0
) -> List[Polygon]:
    """
    Delineate open agricultural fields and barren land plots outside building lots.
    Partitions continuous rural expanses using detected bunds and boundary contours.
    """
    h, w = crop_mask.shape[:2]
    m2_per_px2 = float(meters_per_px) ** 2
    min_px = int(min_area_m2 / m2_per_px2)

    open_land = cv2.bitwise_or(crop_mask, barren_mask)

    # Draw bund divider lines onto open land mask to partition connected fields
    bund_divider_mask = np.zeros((h, w), dtype=np.uint8)
    for seg in bund_segments:
        pt1 = (int(seg[0][0]), int(seg[0][1]))
        pt2 = (int(seg[1][0]), int(seg[1][1]))
        cv2.line(bund_divider_mask, pt1, pt2, 255, 3)

    # Remove bund lines and existing building parcels
    partitioned_land = cv2.bitwise_and(open_land, cv2.bitwise_not(bund_divider_mask))

    # Mask out already allocated building parcels
    occ_mask = np.zeros((h, w), dtype=np.uint8)
    for p in occupied_polys:
        if p.is_valid and not p.is_empty:
            coords = np.array(p.exterior.coords, dtype=np.int32)
            cv2.fillPoly(occ_mask, [coords], 255)

    available_open_land = cv2.bitwise_and(partitioned_land, cv2.bitwise_not(occ_mask))

    # Clean morphological parcels
    clean_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    available_open_land = cv2.morphologyEx(available_open_land, cv2.MORPH_OPEN, clean_k)

    contours, _ = cv2.findContours(available_open_land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rural_plots = []

    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_px:
            # Regularize contour into clean cadastral boundary
            eps = 0.018 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(approx) >= 3:
                p = Polygon(approx)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_valid and not p.is_empty and p.area >= min_px:
                    clipped = p.intersection(roi_box)
                    if not clipped.is_empty and clipped.area >= min_px:
                        if clipped.geom_type == "Polygon":
                            rural_plots.append(clipped)
                        elif clipped.geom_type == "MultiPolygon":
                            for sub_p in clipped.geoms:
                                if sub_p.area >= min_px:
                                    rural_plots.append(sub_p)

    return rural_plots


def extract_discrete_landcover_entities(
    img_bgr: np.ndarray,
    px_to_lonlat_fn: Callable[[float, float], Tuple[float, float]],
    meters_per_px: float = 0.3,
    parcel_features: Optional[List[Dict[str, Any]]] = None,
    building_features: Optional[List[Dict[str, Any]]] = None,
    road_features: Optional[List[Dict[str, Any]]] = None,
    building_mask: Optional[np.ndarray] = None,
    road_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract discrete, identifiable vector entities for:
    - Trees & Orchards (trees.geojson): Real elevated tree crowns (compact, shadow-paired)
    - Agricultural Farm Plots (farms.geojson): Delineated farm fields bounded by bunds
    - Barren Land Plots (barren_land.geojson): Exposed mineral earth and fallow soil
    - Active Green Vegetation (vegetation.geojson): Ground lawns, turf grass, and parks
    
    Returns standard GeoJSON FeatureCollections matching the structure of buildings.geojson,
    with unique IDs, areas in m², sq.ft, guntha, and acres, spectral metrics, and field assessment notes.
    """
    h, w = img_bgr.shape[:2]
    m_per_px = float(meters_per_px) if meters_per_px > 0 else 0.3
    m2_per_px2 = m_per_px ** 2

    # Coordinate mapping bounds
    lon_nw, lat_nw = px_to_lonlat_fn(0, 0)
    lon_se, lat_se = px_to_lonlat_fn(w, h)
    d_lon = (lon_se - lon_nw) if abs(lon_se - lon_nw) > 1e-9 else 1.0
    d_lat = (lat_se - lat_nw) if abs(lat_se - lat_nw) > 1e-9 else 1.0

    def geo_to_px(lon, lat):
        return int((lon - lon_nw) / d_lon * w), int((lat - lat_nw) / d_lat * h)

    # 1. Rasterize building mask if features provided and mask not given
    if building_mask is None and building_features:
        building_mask = np.zeros((h, w), dtype=np.uint8)
        for bf in building_features:
            geom = bf.get("geometry", {})
            gtype = geom.get("type", "")
            coords = geom.get("coordinates", [])
            if gtype == "Polygon" and coords:
                pts = np.array([geo_to_px(pt[0], pt[1]) for pt in coords[0]], dtype=np.int32)
                cv2.fillPoly(building_mask, [pts], 255)
            elif gtype == "MultiPolygon" and coords:
                for poly in coords:
                    pts = np.array([geo_to_px(pt[0], pt[1]) for pt in poly[0]], dtype=np.int32)
                    cv2.fillPoly(building_mask, [pts], 255)

    # 2. Rasterize road mask if features provided and mask not given
    if road_mask is None and road_features:
        road_mask = np.zeros((h, w), dtype=np.uint8)
        for rf in road_features:
            geom = rf.get("geometry", {})
            coords = geom.get("coordinates", [])
            if geom.get("type") == "LineString" and len(coords) >= 2:
                pts = np.array([geo_to_px(pt[0], pt[1]) for pt in coords], dtype=np.int32)
                cv2.polylines(road_mask, [pts], False, 255, max(8, int(6.0 / m_per_px)))

    # Compute spectral indices with building and road masking
    indices = compute_visible_vegetation_and_soil_indices(
        img_bgr=img_bgr,
        building_mask=building_mask,
        road_mask=road_mask,
        meters_per_px=m_per_px
    )
    exg = indices["exg"]
    vari = indices["vari"]
    sti = indices["sti"]
    tree_mask = indices["tree_mask"]
    crop_mask = indices["crop_mask"]
    barren_mask = indices["barren_mask"]
    veg_mask = indices["veg_mask"]
    std15 = indices["std15"]

    # Detect bund lines to partition agricultural fields
    bund_segments = detect_agricultural_bunds(img_bgr, crop_mask, barren_mask, meters_per_px=m_per_px)

    # Helper: Find parcel ULPIN for a given polygon centroid
    def get_parent_ulpin(centroid_pt: Point) -> str:
        if not parcel_features:
            return ""
        for pf in parcel_features:
            geom = shape(pf["geometry"])
            if geom.contains(centroid_pt):
                return pf["properties"].get("ulpin", "")
        return ""

    # ── 1. Discrete Forest Zones (Dense Contiguous Canopy Patches) ───────────
    # Forest = large contiguous wooded / multi-tree canopy zones.
    # Strategy: merge adjacent tree-canopy pixels with a large morphological closing
    # kernel so individual crowns fuse into coherent forest patches. Apply a minimum
    # area threshold (> ~200 m²) so isolated ornamental trees don't qualify as forest.
    #
    # Requirements:
    #   - Pixels must be in tree_mask (high-texture, shadow-paired, elevated vegetation)
    #   - Blob area >= min_forest_m2 (200 m²) after closing
    #   - Mean ExG >= 0.06 (dense healthy canopy)
    #   - High std15 texture (> 10) — canopy roughness, not flat lawn

    min_forest_m2 = 200.0
    min_forest_px = max(50, int(min_forest_m2 / m2_per_px2))

    # Large closing kernel (25×25 px at 0.3 m/px ≈ 7.5 m) fuses adjacent crowns
    k_forest_close = min(25, max(7, int(6.0 / m_per_px)))
    if k_forest_close % 2 == 0:
        k_forest_close += 1
    forest_merged = cv2.morphologyEx(
        tree_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_forest_close, k_forest_close))
    )
    # Clean tiny noise blobs left from the closing
    forest_merged = cv2.morphologyEx(
        forest_merged,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    f_contours_forest, _ = cv2.findContours(forest_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    forest_features = []
    forest_reports = {}
    fst_id = 1

    for c in f_contours_forest:
        area_px = cv2.contourArea(c)
        if area_px < min_forest_px:
            continue

        # Compute mask for this blob
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)

        # Must have meaningful canopy texture (not flat lawn)
        mean_std15 = float(np.mean(std15[c_mask > 0]))
        if mean_std15 < 8.0:
            continue

        mean_exg = round(float(np.mean(exg[c_mask > 0])), 3)
        if mean_exg < 0.04:
            continue

        # Smoothed polygon boundary
        eps = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue

        geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in approx]
        if geo_pts[0] != geo_pts[-1]:
            geo_pts.append(geo_pts[0])
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid:
            poly_geo = poly_geo.buffer(0)
        if poly_geo.is_empty or poly_geo.area <= 0:
            continue

        area_m2 = round(area_px * m2_per_px2, 1)
        mean_vari = round(float(np.mean(vari[c_mask > 0])), 3)
        # Forest density: fraction of closing blob that was original tree_mask pixels
        tree_px_in_blob = int(np.count_nonzero(cv2.bitwise_and(tree_mask, c_mask)))
        forest_density = round(tree_px_in_blob / max(1, int(area_px)), 2)
        ulpin = get_parent_ulpin(poly_geo.centroid)

        forest_type = (
            "Dense Forest / Closed Canopy" if forest_density >= 0.60 and mean_exg > 0.08
            else "Mixed Forest / Sparse Canopy" if forest_density >= 0.35
            else "Open Woodland / Agro-Forestry"
        )

        props = {
            "id": fst_id,
            "uid": f"FOREST-{fst_id:04d}",
            "name": f"Forest Zone #{fst_id}",
            "type": "forest_zone",
            "category": "Forest / Dense Canopy",
            "area_m2": area_m2,
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "area_guntha": round(area_m2 / 101.17, 3),
            "area_acres": round(area_m2 / 4046.86, 4),
            "perimeter_m": round(cv2.arcLength(c, True) * m_per_px, 1),
            "mean_exg": mean_exg,
            "mean_vari": mean_vari,
            "canopy_texture": round(mean_std15, 1),
            "forest_density": forest_density,
            "forest_type": forest_type,
            "parcel_ulpin": ulpin,
            "alerts": [
                {"type": "FOREST_COVER", "msg": f"Dense forested canopy zone ({forest_type}, area={area_m2} m², density={forest_density:.0%}).", "citation": "Forest Survey of India / LULC Classification"}
            ]
        }
        forest_features.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        forest_reports[f"FOREST-{fst_id:04d}"] = (
            f"CADASTRAAI -- FOREST / DENSE CANOPY VERIFICATION NOTE\n"
            f"Entity Ref: FOREST-{fst_id:04d}\n"
            f"Category: Forest Zone / Dense Tree Canopy\n"
            f"Parent Parcel ULPIN: {ulpin or 'N/A'}\n"
            f"Canopy Area: {area_m2} m2 ({round(area_m2 / 101.17, 3)} Guntha / {round(area_m2/4046.86, 4)} Acres)\n"
            f"Perimeter: {props['perimeter_m']} m\n"
            f"Forest Type: {forest_type}\n"
            f"Photosynthetic Vitality (ExG): {mean_exg:.3f} | Canopy VARI: {mean_vari:.3f}\n"
            f"Forest Density Index: {forest_density:.2f} | Canopy Texture: {mean_std15:.1f}\n\n"
            f"SURVEY ASSESSMENT:\n"
            f"Dense multi-crown canopy cluster identified via morphological canopy fusion,\n"
            f"high ExG index, and elevated canopy texture roughness.\n"
            f"Classified as forested land under LULC / FSI canopy cover standards."
        )
        fst_id += 1

    # ── 2. Discrete Agricultural Farm Fields ─────────────────────────────────
    bund_mask = np.zeros((h, w), dtype=np.uint8)
    for seg in bund_segments:
        cv2.line(bund_mask, (int(seg[0][0]), int(seg[0][1])), (int(seg[1][0]), int(seg[1][1])), 255, 3)
    part_crops = cv2.bitwise_and(crop_mask, cv2.bitwise_not(bund_mask))
    part_crops = cv2.morphologyEx(part_crops, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    min_farm_px = max(12, int(60.0 / m2_per_px2))
    f_contours, _ = cv2.findContours(part_crops, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    f_valid = [c for c in f_contours if cv2.contourArea(c) >= min_farm_px]
    f_valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    farm_features = []
    farm_reports = {}
    for fid, c in enumerate(f_valid, 1):
        area_px = cv2.contourArea(c)
        area_m2 = round(area_px * m2_per_px2, 1)
        eps = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in approx]
        if geo_pts[0] != geo_pts[-1]:
            geo_pts.append(geo_pts[0])
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid:
            poly_geo = poly_geo.buffer(0)
        if poly_geo.is_empty or poly_geo.area <= 0:
            continue

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        mean_exg = round(float(np.mean(exg[c_mask > 0])), 3)
        mean_vari = round(float(np.mean(vari[c_mask > 0])), 3)
        ulpin = get_parent_ulpin(poly_geo.centroid)

        props = {
            "id": fid,
            "uid": f"FARM-{fid:04d}",
            "name": f"Farm Field #{fid}",
            "type": "farm_plot",
            "category": "Agricultural / Cropland",
            "area_m2": area_m2,
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "area_guntha": round(area_m2 / 101.17, 3),
            "area_acres": round(area_m2 / 4046.86, 4),
            "perimeter_m": round(cv2.arcLength(c, True) * m_per_px, 1),
            "mean_exg": mean_exg,
            "mean_vari": mean_vari,
            "crop_canopy_index": round(float(np.count_nonzero(veg_mask[c_mask > 0])) / max(1, area_px), 2),
            "cultivation_status": "Active Cropland / Peak Vigor" if mean_vari > 0.12 else "Cultivated Agricultural Plot",
            "parcel_ulpin": ulpin,
            "alerts": [
                {"type": "CROP_VIGOR", "msg": f"Cultivated cropland verified (ExG={mean_exg:.2f}, VARI={mean_vari:.2f}).", "citation": "National Remote Sensing Cropland Classification"}
            ]
        }
        farm_features.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        farm_reports[f"FARM-{fid:04d}"] = (
            f"CADASTRAAI -- AGRICULTURAL CROPLAND VERIFICATION NOTE\n"
            f"Entity Ref: FARM-{fid:04d}\n"
            f"Category: Agricultural Farm Field / Cropland Plot\n"
            f"Parent Parcel ULPIN: {ulpin or 'N/A'}\n"
            f"Cultivable Area: {area_m2} m2 ({round(area_m2 / 101.17, 3)} Guntha / {round(area_m2/4046.86, 4)} Acres)\n"
            f"Perimeter: {props['perimeter_m']} m\n"
            f"Crop Canopy Index (CCI): {props['crop_canopy_index']:.2f}\n"
            f"Photosynthetic Vitality (ExG): {mean_exg:.3f} | Chlorophyll (VARI): {mean_vari:.3f}\n\n"
            f"SURVEY ASSESSMENT:\n"
            f"Active seasonal cultivation verified via visible spectral reflectance.\n"
            f"Parcel boundaries aligned with agricultural bund ridges and access tracks."
        )

    # ── 3. Discrete Barren Land / Fallow Earth Plots ─────────────────────────
    part_barren = cv2.bitwise_and(barren_mask, cv2.bitwise_not(bund_mask))
    part_barren = cv2.morphologyEx(part_barren, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    min_barren_px = max(10, int(50.0 / m2_per_px2))
    b_contours, _ = cv2.findContours(part_barren, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b_valid = [c for c in b_contours if cv2.contourArea(c) >= min_barren_px]
    b_valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    barren_features = []
    barren_reports = {}
    for bid, c in enumerate(b_valid, 1):
        area_px = cv2.contourArea(c)
        area_m2 = round(area_px * m2_per_px2, 1)
        eps = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in approx]
        if geo_pts[0] != geo_pts[-1]:
            geo_pts.append(geo_pts[0])
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid:
            poly_geo = poly_geo.buffer(0)
        if poly_geo.is_empty or poly_geo.area <= 0:
            continue

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        mean_sti = round(float(np.mean(sti[c_mask > 0])), 3)
        mean_exg = round(float(np.mean(exg[c_mask > 0])), 3)
        ulpin = get_parent_ulpin(poly_geo.centroid)

        props = {
            "id": bid,
            "uid": f"BARREN-{bid:04d}",
            "name": f"Barren Land #{bid}",
            "type": "barren_land",
            "category": "Barren Land / Bare Soil",
            "area_m2": area_m2,
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "area_guntha": round(area_m2 / 101.17, 3),
            "area_acres": round(area_m2 / 4046.86, 4),
            "perimeter_m": round(cv2.arcLength(c, True) * m_per_px, 1),
            "soil_tone_index": mean_sti,
            "mean_exg": mean_exg,
            "land_condition": "Dry Bare Soil / Fallow Ground" if mean_sti > 0.18 else "Exposed Mineral Earth / Sparse Soil",
            "parcel_ulpin": ulpin,
            "alerts": [
                {"type": "SOIL_EXPOSURE", "msg": f"Exposed bare ground confirmed via Soil Tone Index ({mean_sti:.2f}). No standing crops.", "citation": "Land Use / Land Cover (LULC) Classification Standards"}
            ]
        }
        barren_features.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        barren_reports[f"BARREN-{bid:04d}"] = (
            f"CADASTRAAI -- BARREN / FALLOW LAND VERIFICATION NOTE\n"
            f"Entity Ref: BARREN-{bid:04d}\n"
            f"Category: Barren Land / Fallow Earth Plot\n"
            f"Parent Parcel ULPIN: {ulpin or 'N/A'}\n"
            f"Barren Land Area: {area_m2} m2 ({round(area_m2 / 101.17, 3)} Guntha / {round(area_m2/4046.86, 4)} Acres)\n"
            f"Perimeter: {props['perimeter_m']} m\n"
            f"Soil Tone Index (STI): {mean_sti:.3f}\n"
            f"Vegetation Residual (ExG): {mean_exg:.3f}\n\n"
            f"SURVEY ASSESSMENT:\n"
            f"Absence of active vegetative cover and presence of high soil spectral reflectance.\n"
            f"Classified under revenue records as Banjar / Fallow / Open Uncultivated Land."
        )

    # ── 4. Discrete Green Vegetation Zones (Ground Lawns & Open Greenery) ────
    ground_veg_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(tree_mask))
    min_veg_px = max(10, int(30.0 / m2_per_px2))
    v_contours, _ = cv2.findContours(ground_veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_valid = [c for c in v_contours if cv2.contourArea(c) >= min_veg_px]
    v_valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    veg_features = []
    veg_reports = {}
    for vid, c in enumerate(v_valid, 1):
        area_px = cv2.contourArea(c)
        area_m2 = round(area_px * m2_per_px2, 1)
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in approx]
        if geo_pts[0] != geo_pts[-1]:
            geo_pts.append(geo_pts[0])
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid:
            poly_geo = poly_geo.buffer(0)
        if poly_geo.is_empty or poly_geo.area <= 0:
            continue

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        mean_exg = round(float(np.mean(exg[c_mask > 0])), 3)
        mean_vari = round(float(np.mean(vari[c_mask > 0])), 3)
        ulpin = get_parent_ulpin(poly_geo.centroid)

        props = {
            "id": vid,
            "uid": f"VEG-{vid:04d}",
            "name": f"Green Vegetation #{vid}",
            "type": "vegetation",
            "category": "Active Green Vegetation",
            "area_m2": area_m2,
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "area_acres": round(area_m2 / 4046.86, 4),
            "mean_exg": mean_exg,
            "mean_vari": mean_vari,
            "parcel_ulpin": ulpin,
            "alerts": []
        }
        veg_features.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        veg_reports[f"VEG-{vid:04d}"] = (
            f"CADASTRAAI -- VEGETATION ZONE NOTE\n"
            f"Entity Ref: VEG-{vid:04d}\n"
            f"Category: General Green Cover (Grassland, Lawn, Shrub)\n"
            f"Parent Parcel ULPIN: {ulpin or 'N/A'}\n"
            f"Area: {area_m2} m2\n"
            f"ExG: {mean_exg:.3f} | VARI: {mean_vari:.3f}\n"
        )

    # ── 5. Agricultural Bund Feature Collection ──────────────────────────────
    bund_features = []
    for bid, seg in enumerate(bund_segments, 1):
        coords = [px_to_lonlat_fn(float(seg[0][0]), float(seg[0][1])), px_to_lonlat_fn(float(seg[1][0]), float(seg[1][1]))]
        bund_features.append({
            "type": "Feature",
            "properties": {"id": bid, "uid": f"BUND-{bid:04d}", "type": "agricultural_bund", "name": f"Agricultural Bund #{bid}"},
            "geometry": {"type": "LineString", "coordinates": coords}
        })

    # Summary
    total_m2 = round(h * w * m2_per_px2, 1)
    forest_m2 = round(sum(f["properties"]["area_m2"] for f in forest_features), 1)
    farm_m2 = round(sum(f["properties"]["area_m2"] for f in farm_features), 1)
    barren_m2 = round(sum(f["properties"]["area_m2"] for f in barren_features), 1)
    veg_m2 = round(sum(f["properties"]["area_m2"] for f in veg_features), 1)

    all_reports = {}
    all_reports.update(forest_reports)
    all_reports.update(farm_reports)
    all_reports.update(barren_reports)
    all_reports.update(veg_reports)

    return {
        "forest_fc": {"type": "FeatureCollection", "features": forest_features},
        "farms_fc": {"type": "FeatureCollection", "features": farm_features},
        "barren_fc": {"type": "FeatureCollection", "features": barren_features},
        "vegetation_fc": {"type": "FeatureCollection", "features": veg_features},
        "bunds_fc": {"type": "FeatureCollection", "features": bund_features},
        "reports": all_reports,
        "summary": {
            "total_survey_area_m2": total_m2,
            "forest_zones_count": len(forest_features),
            "forest_canopy_area_m2": forest_m2,
            "forest_canopy_pct": round((forest_m2 / total_m2) * 100.0, 1) if total_m2 > 0 else 0.0,
            "farm_plots_count": len(farm_features),
            "farm_plots_area_m2": farm_m2,
            "farm_plots_pct": round((farm_m2 / total_m2) * 100.0, 1) if total_m2 > 0 else 0.0,
            "barren_land_count": len(barren_features),
            "barren_land_area_m2": barren_m2,
            "barren_land_pct": round((barren_m2 / total_m2) * 100.0, 1) if total_m2 > 0 else 0.0,
            "total_green_cover_m2": veg_m2,
            "total_green_cover_pct": round((veg_m2 / total_m2) * 100.0, 1) if total_m2 > 0 else 0.0,
        }
    }

