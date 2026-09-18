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
from shapely.geometry import Polygon, MultiPolygon, box
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
