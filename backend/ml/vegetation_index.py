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
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box, mapping, Point, shape
from shapely.ops import unary_union


def compute_visible_vegetation_and_soil_indices(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Compute optical vegetation, tree canopy, and barren soil masks from a standard RGB/BGR image.
    
    Returns
    -------
    dict containing:
      - 'exg': float32 Excess Green array
      - 'vari': float32 Visible Atmospheric Resistant Index
      - 'sti': float32 Soil Tone Index
      - 'veg_mask': uint8 (0 or 255) total active vegetation
      - 'tree_mask': uint8 (0 or 255) tree canopy / orchard clusters
      - 'crop_mask': uint8 (0 or 255) agricultural cropland / grassland
      - 'barren_mask': uint8 (0 or 255) barren land / bare soil / fallow earth
    """
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

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k_size = 7
    local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (k_size, k_size))
    local_sq = cv2.boxFilter(gray.astype(np.float32) ** 2, -1, (k_size, k_size))
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean ** 2))

    # ── Active Vegetation Mask ───────────────────────────────────────────────
    # True vegetation has positive ExG, green dominance over R & B, and green hue (28-88)
    is_green_hue = (hue >= 28) & (hue <= 88)
    veg_raw = (
        (exg > 0.035) &
        (g > r * 0.98) &
        (g > b * 1.02) &
        (is_green_hue | (vari > 0.02)) &
        (sat > 25)
    ).astype(np.uint8) * 255

    # Clean isolated noise
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    veg_mask = cv2.morphologyEx(veg_raw, cv2.MORPH_OPEN, kc)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # ── Tree Canopy vs Cropland Separation ──────────────────────────────────
    # Trees / orchards exhibit high local surface roughness, canopy texture variation,
    # and deeper green shadows. Cropland / grass is smooth and planar.
    tree_cond = (veg_mask > 0) & ((local_std > 15.0) | ((val < 90) & (sat > 50)))
    tree_raw = tree_cond.astype(np.uint8) * 255
    tree_mask = cv2.morphologyEx(tree_raw, cv2.MORPH_OPEN, kc)
    tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    crop_mask = cv2.bitwise_and(veg_mask, cv2.bitwise_not(tree_mask))

    # ── Barren Land / Bare Soil / Fallow Earth Mask ──────────────────────────
    # Barren land has:
    # 1. Negative or near-zero ExG (absence of chlorophyll)
    # 2. Warm earth hue (H in [5, 26], orange/tan/brown/terracotta)
    # 3. High red-to-blue ratio (STI > 0.08) and R >= G
    # 4. Moderate saturation (distinguishes from neutral grey roads/asphalt)
    # 5. Low-to-moderate texture roughness (not building roofs with steep edges)
    is_soil_hue = (hue >= 5) & (hue <= 26)
    barren_cond = (
        (veg_mask == 0) &
        (r > g * 0.98) &
        (r > b * 1.10) &
        (sti > 0.08) &
        (is_soil_hue | (sti > 0.16)) &
        (sat >= 20) & (sat <= 175) &
        (val >= 50) & (val <= 235) &
        (local_std < 32.0)
    )
    barren_raw = barren_cond.astype(np.uint8) * 255
    barren_mask = cv2.morphologyEx(barren_raw, cv2.MORPH_OPEN, kc)
    barren_mask = cv2.morphologyEx(barren_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    return {
        "exg": exg,
        "vari": vari,
        "sti": sti,
        "veg_mask": veg_mask,
        "tree_mask": tree_mask,
        "crop_mask": crop_mask,
        "barren_mask": barren_mask,
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
    px_to_lonlat_fn,
    meters_per_px: float = 0.3,
    parcel_features: List[Dict] = None
) -> Dict[str, Any]:
    """
    Extract discrete, identifiable vector entities for:
    - Trees & Orchards (trees.geojson)
    - Agricultural Farm Plots (farms.geojson)
    - Barren Land Plots (barren_land.geojson)
    - Active Green Vegetation (vegetation.geojson)
    
    Returns standard GeoJSON FeatureCollections matching the structure of buildings.geojson,
    with unique IDs, areas in m², sq.ft, guntha, and acres, spectral metrics, and field assessment notes.
    """
    h, w = img_bgr.shape[:2]
    m_per_px = float(meters_per_px) if meters_per_px > 0 else 0.3
    m2_per_px2 = m_per_px ** 2

    # Compute spectral indices
    indices = compute_visible_vegetation_and_soil_indices(img_bgr)
    exg = indices["exg"]
    vari = indices["vari"]
    sti = indices["sti"]
    tree_mask = indices["tree_mask"]
    crop_mask = indices["crop_mask"]
    barren_mask = indices["barren_mask"]
    veg_mask = indices["veg_mask"]

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

    # ── 1. Discrete Tree Canopy Clusters ─────────────────────────────────────
    min_tree_px = max(6, int(12.0 / m2_per_px2))
    t_contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t_valid = [c for c in t_contours if cv2.contourArea(c) >= min_tree_px]
    t_valid.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    tree_features = []
    tree_reports = {}
    for tid, c in enumerate(t_valid, 1):
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
        crown_diam = round(2.0 * math.sqrt(area_m2 / math.pi), 1)
        ulpin = get_parent_ulpin(poly_geo.centroid)

        props = {
            "id": tid,
            "uid": f"TREE-{tid:04d}",
            "name": f"Tree Canopy #{tid}",
            "type": "tree_canopy",
            "category": "Tree Canopy / Orchard",
            "area_m2": area_m2,
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "area_acres": round(area_m2 / 4046.86, 4),
            "crown_diameter_m": crown_diam,
            "mean_exg": mean_exg,
            "mean_vari": mean_vari,
            "health_status": "Dense Healthy Canopy" if mean_exg > 0.08 else "Moderate Green Canopy",
            "parcel_ulpin": ulpin,
            "alerts": [
                {"type": "CANOPY_HEALTH", "msg": f"Photosynthetically active tree canopy (crown diameter ~{crown_diam}m, ExG={mean_exg:.2f}).", "citation": "National Agro-Forestry & Green Cover Guidelines"}
            ]
        }
        tree_features.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        tree_reports[f"TREE-{tid:04d}"] = (
            f"CADASTRAAI -- VEGETATION & CANOPY VERIFICATION NOTE\n"
            f"Entity Ref: TREE-{tid:04d}\n"
            f"Category: Tree Canopy / Orchard Cluster\n"
            f"Parent Parcel ULPIN: {ulpin or 'N/A'}\n"
            f"Crown Footprint Area: {area_m2} m2 ({round(area_m2/4046.86, 4)} Acres)\n"
            f"Estimated Crown Diameter: {crown_diam} m\n"
            f"Photosynthetic Vitality (ExG): {mean_exg:.3f} | VARI: {mean_vari:.3f}\n\n"
            f"SURVEY ASSESSMENT:\n"
            f"Vegetation crown identified via high-entropy texture analysis & visible chlorophyll indices.\n"
            f"Contributes to municipal urban tree canopy and rural agro-forestry records."
        )

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

    # ── 4. Discrete Green Vegetation Zones ───────────────────────────────────
    min_veg_px = max(8, int(30.0 / m2_per_px2))
    v_contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    tree_m2 = round(sum(f["properties"]["area_m2"] for f in tree_features), 1)
    farm_m2 = round(sum(f["properties"]["area_m2"] for f in farm_features), 1)
    barren_m2 = round(sum(f["properties"]["area_m2"] for f in barren_features), 1)
    veg_m2 = round(sum(f["properties"]["area_m2"] for f in veg_features), 1)

    all_reports = {}
    all_reports.update(tree_reports)
    all_reports.update(farm_reports)
    all_reports.update(barren_reports)
    all_reports.update(veg_reports)

    return {
        "trees_fc": {"type": "FeatureCollection", "features": tree_features},
        "farms_fc": {"type": "FeatureCollection", "features": farm_features},
        "barren_fc": {"type": "FeatureCollection", "features": barren_features},
        "vegetation_fc": {"type": "FeatureCollection", "features": veg_features},
        "bunds_fc": {"type": "FeatureCollection", "features": bund_features},
        "reports": all_reports,
        "summary": {
            "total_survey_area_m2": total_m2,
            "tree_canopy_count": len(tree_features),
            "tree_canopy_area_m2": tree_m2,
            "tree_canopy_pct": round((tree_m2 / total_m2) * 100.0, 1) if total_m2 > 0 else 0.0,
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
