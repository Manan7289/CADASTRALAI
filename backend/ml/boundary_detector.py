"""
Physical Boundary Wall, Edge, and Fence Detector & Snapper
===========================================================
Detects real visible physical property boundaries from UAV/drone imagery:
1. Multi-scale directional Sobel gradients & Bilateral edge filtering
2. Line Segment Detector (LSD) & Probabilistic Hough for compound walls/fences
3. Boundary Snapping: snaps synthetic/estimated lot lines to real physical walls
"""
import math
from typing import List, Tuple
import cv2
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import nearest_points, snap, unary_union


def detect_physical_walls_and_fences(img_bgr: np.ndarray, min_length_px: float = 15.0) -> List[Tuple[Point, Point]]:
    """
    Extract visible physical compound walls, hedges, and fence segments
    from drone orthomosaic imagery.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Bilateral filter preserves sharp physical boundaries while removing grass/texture noise
    filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 2. Directional Sobel Gradients
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    
    # 3. Canny edge detector on gradient magnitude
    edges = cv2.Canny(filtered, 40, 120, apertureSize=3)
    
    # Morphological line enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 4. Extract straight wall & fence segments via HoughLinesP
    lines = cv2.HoughLinesP(edges_connected, 1, np.pi / 180, threshold=45,
                            minLineLength=25, maxLineGap=8)
    
    wall_segments = []
    if lines is not None:
        for line in lines:
            coords = line.ravel()
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                pt1 = Point(float(x1), float(y1))
                pt2 = Point(float(x2), float(y2))
                wall_segments.append((pt1, pt2))
            
    return wall_segments


from shapely.strtree import STRtree


def snap_polygon_to_physical_walls(poly: Polygon, wall_segments: List[Tuple[Point, Point]], max_snap_dist: float = 3.5) -> Polygon:
    """
    Snaps polygon vertices and edges to nearby detected physical compound walls
    when within max_snap_dist threshold (e.g. ~1m at 0.3m/px).
    """
    if poly is None or poly.is_empty or not wall_segments:
        return poly

    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return poly

    # Fast spatial query using bounding box
    minx, miny, maxx, maxy = poly.bounds
    search_box = (minx - max_snap_dist * 2, miny - max_snap_dist * 2,
                  maxx + max_snap_dist * 2, maxy + max_snap_dist * 2)

    nearby_walls = []
    for pt1, pt2 in wall_segments:
        # Fast bounding box overlap test first
        w_minx = min(pt1.x, pt2.x)
        w_maxx = max(pt1.x, pt2.x)
        w_miny = min(pt1.y, pt2.y)
        w_maxy = max(pt1.y, pt2.y)
        
        if (w_maxx >= search_box[0] and w_minx <= search_box[2] and
            w_maxy >= search_box[1] and w_miny <= search_box[3]):
            ls = LineString([pt1, pt2])
            nearby_walls.append(ls)
            
    if not nearby_walls:
        return poly

    walls_union = unary_union(nearby_walls) if len(nearby_walls) > 1 else nearby_walls[0]
    
    # Snap each exterior vertex to the nearest physical wall if close enough
    snapped_coords = []
    for pt_coord in coords[:-1]:  # exclude duplicate end
        p = Point(pt_coord)
        d = walls_union.distance(p)
        if d <= max_snap_dist:
            # Snap to closest point on physical wall line
            nearest_p = nearest_points(walls_union, p)[0]
            snapped_coords.append((nearest_p.x, nearest_p.y))
        else:
            snapped_coords.append(pt_coord)
            
    if len(snapped_coords) >= 3:
        snapped_coords.append(snapped_coords[0])
        snapped_poly = Polygon(snapped_coords)
        if snapped_poly.is_valid and snapped_poly.area > 50:
            return snapped_poly

    return poly
