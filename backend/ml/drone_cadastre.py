"""AI-Based Drone Cadastral Feature Extraction & Land Parcel Delineator.

Directly extracts from high-resolution UAV/drone imagery:
1. Physical Boundary Walls & Fences (property lines, compound walls)
2. Road & Access Pathway Corridors (street network)
3. Building Rooftops / Plinth Footprints (regularized polygons)
4. Enclosed Land Parcels via Planar Graph & Watershed Partitioning
   (Eliminates synthetic Voronoi heuristics).
"""
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

MIN_PARCEL_M2 = 30.0
MAX_PARCEL_M2 = 25000.0
MIN_BUILDING_M2 = 20.0
MAX_BUILDING_M2 = 3500.0


def extract_drone_features(img_arr: np.ndarray, meters_per_px: float) -> Dict:
    """Extract physical boundary walls, roads, buildings, and delineated parcels
    directly from drone aerial imagery."""
    h, w = img_arr.shape[:2]
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY) if len(img_arr.shape) == 3 else img_arr

    # 1. Multi-scale Edge & Boundary Wall Detection
    # Drone boundary walls appear as high-gradient ridges separating courtyards/plots
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

    # Adaptive edge thresholding for boundary walls and fences
    wall_edges = cv2.adaptiveThreshold(mag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, -4)
    # Morphological line closing to connect wall segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask = cv2.morphologyEx(wall_edges, cv2.MORPH_CLOSE, close_kernel)

    # 2. Road Network & Access Corridor Extraction
    # Roads have low local standard deviation and moderate luminance
    local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (15, 15))
    local_sq_mean = cv2.boxFilter((gray.astype(np.float32)) ** 2, -1, (15, 15))
    local_var = np.maximum(0, local_sq_mean - local_mean ** 2)
    local_std = np.sqrt(local_var)

    # Road candidate mask: smooth texture (low std) and contiguous
    road_cand = ((local_std < 18) & (gray > 50) & (gray < 220)).astype(np.uint8) * 255
    road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    road_mask = cv2.morphologyEx(road_cand, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # 3. Building Plinth & Rooftop Extraction
    # Buildings have elevated texture contrast, distinct roof tones, and boundary contours
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 4)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    building_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Clean building contours
    min_b_px = max(6, int(MIN_BUILDING_M2 / (meters_per_px ** 2)))
    max_b_px = max(min_b_px + 2, int(MAX_BUILDING_M2 / (meters_per_px ** 2)))

    b_contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings_px = []
    for c in b_contours:
        a = cv2.contourArea(c)
        if min_b_px <= a <= max_b_px:
            # Regularize / orthogonalize building footprint corners
            eps = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(approx) >= 4:
                buildings_px.append(approx)

    # 4. Planar Graph & Watershed Parcel Delineation
    # Combine road network + boundary walls to form the cadastral boundary partition
    combined_dividers = cv2.bitwise_or(wall_mask, road_mask)
    # Dilate dividers slightly to enforce clear separation
    div_kernel = np.ones((3, 3), np.uint8)
    combined_dividers = cv2.dilate(combined_dividers, div_kernel, iterations=1)

    # Invert to get parcel interiors
    interiors = cv2.bitwise_not(combined_dividers)
    # Distance transform creates distinct parcel peaks
    dist = cv2.distanceTransform(interiors, cv2.DIST_L2, 5)
    _, markers_seed = cv2.threshold(dist, 0.25 * dist.max(), 255, cv2.THRESH_BINARY)
    markers_seed = markers_seed.astype(np.uint8)

    # Connected components on interior seeds
    num_markers, markers = cv2.connectedComponents(markers_seed)

    # Watershed segmentation on inverse distance
    # Convert image to 3-channel for cv2.watershed
    ws_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.copy().astype(np.int32)
    cv2.watershed(ws_input, markers_ws)

    # Extract distinct parcel polygons from watershed regions
    min_p_px = max(10, int(MIN_PARCEL_M2 / (meters_per_px ** 2)))
    max_p_px = max(min_p_px + 10, int(MAX_PARCEL_M2 / (meters_per_px ** 2)))

    parcels_px = []
    unique_labels = np.unique(markers_ws)
    for lbl in unique_labels:
        if lbl <= 1:  # 0: boundary line, 1: background border
            continue
        p_mask = (markers_ws == lbl).astype(np.uint8) * 255
        p_contours, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in p_contours:
            a = cv2.contourArea(c)
            if min_p_px <= a <= max_p_px:
                eps = 0.015 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
                if len(approx) >= 3:
                    parcels_px.append(approx)

    # Fallback / Boundary partitioning if watershed produced too few plots
    if len(parcels_px) < 3:
        # Contour extraction on interior spaces
        inter_contours, _ = cv2.findContours(interiors, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in inter_contours:
            a = cv2.contourArea(c)
            if min_p_px <= a <= max_p_px:
                eps = 0.015 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
                if len(approx) >= 3:
                    parcels_px.append(approx)

    # Extract road lines / corridors
    road_contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roads_px = []
    for c in road_contours:
        if cv2.contourArea(c) > (min_p_px / 2):
            eps = 0.02 * cv2.arcLength(c, False)
            approx = cv2.approxPolyDP(c, eps, False).reshape(-1, 2)
            if len(approx) >= 2:
                roads_px.append(approx)

    # Extract wall lines
    wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    walls_px = []
    for c in wall_contours:
        if cv2.arcLength(c, False) > (20.0 / meters_per_px):
            eps = 0.02 * cv2.arcLength(c, False)
            approx = cv2.approxPolyDP(c, eps, False).reshape(-1, 2)
            if len(approx) >= 2:
                walls_px.append(approx)

    return {
        "buildings_px": buildings_px,
        "parcels_px": parcels_px,
        "roads_px": roads_px,
        "walls_px": walls_px,
        "wall_mask": wall_mask,
        "road_mask": road_mask,
    }


def pixel_polygons_to_geospatial(polys_px: List[np.ndarray], px_to_lonlat) -> List[Polygon]:
    """Convert pixel coordinate polygons to valid WGS84 GeoJSON Shapely Polygons."""
    geo_polys = []
    for approx in polys_px:
        ring = [px_to_lonlat(float(px), float(py)) for px, py in approx]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid and poly.area > 0:
            if poly.geom_type == "Polygon":
                geo_polys.append(poly)
            elif poly.geom_type == "MultiPolygon":
                geo_polys.extend([g for g in poly.geoms if g.area > 0])
    return geo_polys
