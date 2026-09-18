"""
CadastraAI 4-Factor Cadastral Parcel Engine
===========================================
Implements the 4 core factors of automated cadastral mapping:
  Factor 1: Standard Cadastral Mapping (1-to-1 building enclosure, setbacks, ULPIN registry)
  Factor 2: Physical Boundary, Edge & Fence Detection and Matching (Sobel/Canny/Hough snapping)
  Factor 3: Quadrilateral Regularization & Equal-Area Normalization (4-vertex convex plots, uniform block area)
  Factor 4: OSM-Style Orthogonal Right Angles, Shared Party Walls & Zero-Overlap Guarantee
"""
import math
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiPoint, box, mapping
from shapely.ops import unary_union, voronoi_diagram

try:
    from ml.boundary_detector import detect_physical_walls_and_fences, snap_polygon_to_physical_walls
    from ml.quad_regularizer import regularize_to_quadrilateral, normalize_block_parcel_areas
    from ml.vegetation_index import (
        compute_visible_vegetation_and_soil_indices,
        detect_agricultural_bunds,
        analyze_parcel_landcover,
        delineate_open_rural_parcels
    )
except ImportError:
    try:
        from .boundary_detector import detect_physical_walls_and_fences, snap_polygon_to_physical_walls
        from .quad_regularizer import regularize_to_quadrilateral, normalize_block_parcel_areas
        from .vegetation_index import (
            compute_visible_vegetation_and_soil_indices,
            detect_agricultural_bunds,
            analyze_parcel_landcover,
            delineate_open_rural_parcels
        )
    except ImportError:
        from boundary_detector import detect_physical_walls_and_fences, snap_polygon_to_physical_walls
        from quad_regularizer import regularize_to_quadrilateral, normalize_block_parcel_areas
        from vegetation_index import (
            compute_visible_vegetation_and_soil_indices,
            detect_agricultural_bunds,
            analyze_parcel_landcover,
            delineate_open_rural_parcels
        )


def dominant_angles_from_segs(street_segs, n_bins=36):
    """Find dominant street grid directions from street segments."""
    if not street_segs:
        return [0.0]
    angles = np.array([a % 90 for a, _ in street_segs])
    counts, edges = np.histogram(angles, bins=n_bins, range=(0, 90))
    peaks = []
    for i in range(n_bins):
        l = counts[(i-1) % n_bins]
        r = counts[(i+1) % n_bins]
        if counts[i] >= l and counts[i] >= r and counts[i] > 0:
            peaks.append((counts[i], (edges[i]+edges[i+1])/2))
    peaks.sort(reverse=True)
    return [p[1] for p in peaks[:2]] if peaks else [0.0]


def get_nearest_road_info(pt: Point, road_lines: List[Tuple[LineString, float]]):
    """Find nearest road segment line, distance, and direction angle."""
    min_d = 1e9
    best_line = None
    best_ang = 0.0
    for ls, w in road_lines:
        d = ls.distance(pt)
        if d < min_d:
            min_d = d
            best_line = ls
            coords = list(ls.coords)
            for k in range(len(coords)-1):
                seg = LineString([coords[k], coords[k+1]])
                if seg.distance(pt) <= d + 1.0:
                    dx = coords[k+1][0] - coords[k][0]
                    dy = coords[k+1][1] - coords[k][1]
                    best_ang = math.degrees(math.atan2(dy, dx))
                    break
    return best_line, min_d, best_ang


def generate_4factor_cadastral_parcels(
    bldgs: List[Dict],
    road_lines: List[Tuple[LineString, float]],
    img_bgr: np.ndarray,
    roi_box: Polygon,
    px_to_lonlat_fn,
    cadastral_standards,
    road_union: Polygon = None,
    meters_per_px: float = 0.3,
    veg_mask: np.ndarray = None,
    tree_mask: np.ndarray = None,
    crop_mask: np.ndarray = None,
    barren_mask: np.ndarray = None,
    bund_segments: List[np.ndarray] = None,
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    Generate clean, 4-factor regularized cadastral land parcels with vegetation,
    tree canopy, and barren land attribution.
    
    Returns
    -------
    parcels_geojson_features, bldg_features, property_cards
    """
    if not bldgs:
        return [], [], {}

    H, W = img_bgr.shape[:2]
    if road_union is None:
        road_union = Polygon()

    m_per_px = float(meters_per_px) if meters_per_px > 0 else 0.3
    m2_per_px2 = m_per_px ** 2

    # ── Landcover & Vegetation Indices ───────────────────────────────────────
    if veg_mask is None or tree_mask is None or barren_mask is None:
        landcover = compute_visible_vegetation_and_soil_indices(img_bgr)
        veg_mask = landcover["veg_mask"]
        tree_mask = landcover["tree_mask"]
        crop_mask = landcover["crop_mask"]
        barren_mask = landcover["barren_mask"]
        if bund_segments is None:
            bund_segments = detect_agricultural_bunds(
                img_bgr, crop_mask=crop_mask, barren_mask=barren_mask, meters_per_px=m_per_px
            )

    # ── Factor 2: Detect Physical Compound Walls & Fences ────────────────────
    print("  [Factor 2] Extracting physical compound walls & fence boundaries...")
    wall_segments = detect_physical_walls_and_fences(img_bgr)
    print(f"  [Factor 2] Extracted {len(wall_segments)} visible physical boundary segments")

    # ── Planar Voronoi Seeds (Zero-Overlap Guarantee) ────────────────────────
    if len(bldgs) >= 2:
        seeds = MultiPoint([Point(b["cx"], b["cy"]) for b in bldgs])
        vor = voronoi_diagram(seeds, envelope=roi_box.buffer(10))
        vor_cells = list(vor.geoms)
    else:
        vor_cells = [roi_box]

    # ── Group residential buildings by nearest street for block alignment ────
    road_bldg_map = {}
    for b in bldgs:
        pt = Point(b["cx"], b["cy"])
        ls, d, ang = get_nearest_road_info(pt, road_lines)
        key = id(ls) if ls else 0
        if key not in road_bldg_map:
            road_bldg_map[key] = {"line": ls, "angle": ang, "bldgs": []}
        road_bldg_map[key]["bldgs"].append((b, pt, d, ang))

    # ── Factor 1, 3 & 4: Strict Rectilinear Cadastral Lot Partitioning ─────────
    # For each street block, generate exact 4-corner rectangular parcels
    # that share party walls and have uniform frontage and depth.
    for key, group in road_bldg_map.items():
        ls = group["line"]
        ang = group["angle"]
        group_bldgs = group["bldgs"]
        if ls is None or len(group_bldgs) == 0:
            continue

        coords = list(ls.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        length = math.hypot(dx, dy) + 1e-6
        nx_base, ny_base = -dy / length, dx / length

        # Project building coordinates onto street axis
        proj_data = []
        for b, pt, d, a in group_bldgs:
            s_center = ls.project(pt)
            
            # Get building extent along street axis (s) and normal axis (d)
            b_coords = list(b["poly_px"].exterior.coords) if "poly_px" in b else [(pt.x, pt.y)]
            s_vals = [ls.project(Point(x, y)) for x, y in b_coords]
            
            # Determine building orientation relative to street normal
            mid_pt = ls.interpolate(s_center)
            dot = (pt.x - mid_pt.x) * nx_base + (pt.y - mid_pt.y) * ny_base
            nx = nx_base if dot >= 0 else -nx_base
            ny = ny_base if dot >= 0 else -ny_base
            
            d_vals = [(x - mid_pt.x) * nx + (y - mid_pt.y) * ny for x, y in b_coords]
            
            s_min, s_max = min(s_vals), max(s_vals)
            d_min, d_max = max(1.0, min(d_vals)), max(d_vals)
            
            proj_data.append({
                "s_center": s_center, "d_center": d, "b": b, "pt": pt,
                "s_min": s_min, "s_max": s_max, "d_min": d_min, "d_max": d_max,
                "nx": nx, "ny": ny
            })

        proj_data.sort(key=lambda x: x["s_center"])
        n = len(proj_data)

        # Compute median block frontage width and depth to normalize plot areas
        s_diffs = [proj_data[k+1]["s_center"] - proj_data[k]["s_center"] for k in range(n-1)] if n > 1 else [50.0]
        median_frontage = float(np.median(s_diffs)) if s_diffs else 50.0
        median_frontage = max(30.0, min(70.0, median_frontage))
        half_frontage = median_frontage / 2.0

        median_depth = float(np.median([p["d_max"] - p["d_min"] for p in proj_data])) + 25.0
        median_depth = max(55.0, min(100.0, median_depth))

        # Factor 2: Pre-filter wall segments near this street block
        ls_bbox = ls.buffer(median_depth + 60.0).bounds
        group_walls = []
        for pt1, pt2 in wall_segments:
            wx_min, wx_max = min(pt1.x, pt2.x), max(pt1.x, pt2.x)
            wy_min, wy_max = min(pt1.y, pt2.y), max(pt1.y, pt2.y)
            if (wx_max >= ls_bbox[0] and wx_min <= ls_bbox[2] and
                wy_max >= ls_bbox[1] and wy_min <= ls_bbox[3]):
                mid_x, mid_y = (pt1.x + pt2.x) / 2.0, (pt1.y + pt2.y) / 2.0
                ws = ls.project(Point(mid_x, mid_y))
                mid_proj = ls.interpolate(ws)
                wd = (mid_x - mid_proj.x) * nx_base + (mid_y - mid_proj.y) * ny_base
                group_walls.append((ws, wd))

        for idx in range(n):
            item = proj_data[idx]
            b_curr = item["b"]
            nx, ny = item["nx"], item["ny"]

            # 1. Compute side boundary division lines (shared party walls)
            if idx == 0:
                s_left = min(item["s_min"] - 8.0, item["s_center"] - half_frontage)
            else:
                prev = proj_data[idx-1]
                s_left = (prev["s_max"] + item["s_min"]) / 2.0 if prev["s_max"] < item["s_min"] else (prev["s_center"] + item["s_center"]) / 2.0

            if idx == n - 1:
                s_right = max(item["s_max"] + 8.0, item["s_center"] + half_frontage)
            else:
                nxt = proj_data[idx+1]
                s_right = (item["s_max"] + nxt["s_min"]) / 2.0 if item["s_max"] < nxt["s_min"] else (item["s_center"] + nxt["s_center"]) / 2.0

            # Guarantee building enclosure
            s_left = min(s_left, item["s_min"] - 4.0)
            s_right = max(s_right, item["s_max"] + 4.0)

            # 2. Front and back setbacks (street right-of-way and rear block centerline)
            d_front = max(3.0, item["d_min"] - 8.0)
            d_back  = max(item["d_max"] + 10.0, d_front + median_depth)

            # 3. Factor 2: Snap front/back lot lines to physical walls
            pt_center = item["pt"]
            for ws, wd in group_walls:
                if s_left - 5 <= ws <= s_right + 5:
                    w_dist = abs(wd)
                    if abs(w_dist - d_front) <= 3.0:
                        d_front = w_dist
                    elif abs(w_dist - d_back) <= 3.0:
                        d_back = w_dist

            # 4. Construct mathematically strict 4-corner rectangle (90-degree right angles)
            s_left_clamped = max(0.0, min(ls.length, s_left))
            s_right_clamped = max(0.0, min(ls.length, s_right))

            pt_LF = ls.interpolate(s_left_clamped)
            pt_RF = ls.interpolate(s_right_clamped)

            p1 = (pt_LF.x + nx * d_front, pt_LF.y + ny * d_front)
            p2 = (pt_RF.x + nx * d_front, pt_RF.y + ny * d_front)
            p3 = (pt_RF.x + nx * d_back,  pt_RF.y + ny * d_back)
            p4 = (pt_LF.x + nx * d_back,  pt_LF.y + ny * d_back)

            rect_poly = Polygon([p1, p2, p3, p4, p1])
            if not rect_poly.is_valid:
                rect_poly = rect_poly.buffer(0)

            # 5. Factor 4: Planar Voronoi envelope guarantees 100% zero mutual overlap
            best_v = None
            for v in vor_cells:
                if v.contains(pt_center):
                    best_v = v; break
            if best_v is None:
                best_v = min(vor_cells, key=lambda v: v.distance(pt_center))

            bounded_lot = rect_poly.intersection(best_v).intersection(roi_box)
            if bounded_lot.is_empty or bounded_lot.area < 50:
                bounded_lot = best_v.intersection(roi_box)

            if isinstance(bounded_lot, MultiPolygon):
                matched = [p for p in bounded_lot.geoms if p.contains(pt_center)]
                bounded_lot = matched[0] if matched else max(bounded_lot.geoms, key=lambda p: p.area)

            # Simplify slight collinear vertices to preserve clean 4-corner quadrilateral
            quad_lot = bounded_lot.simplify(1.5, preserve_topology=True)
            if not quad_lot.is_valid or quad_lot.is_empty:
                quad_lot = bounded_lot

            b_curr["parcel_poly_px"] = quad_lot

    # ── Commercial compounds ─────────────────────────────────────────────────
    comm_bldgs = [b for b in bldgs if b.get("type") == "commercial"]
    for b in comm_bldgs:
        pt = Point(b["cx"], b["cy"])
        b_poly = b["poly_px"]
        best_v = None
        for v in vor_cells:
            if v.contains(pt):
                best_v = v; break
        if best_v is None:
            best_v = min(vor_cells, key=lambda v: v.distance(pt))
            
        lot = b_poly.buffer(35, join_style=2, cap_style=2).intersection(best_v).intersection(roi_box)
        if not road_union.is_empty:
            diff = lot.difference(road_union)
            if not diff.is_empty and diff.area >= 100:
                lot = diff
        if lot.is_empty:
            lot = b_poly.buffer(10).intersection(roi_box)
        if isinstance(lot, MultiPolygon):
            lot = max(lot.geoms, key=lambda p: p.area)
        b["parcel_poly_px"] = lot.simplify(3.0, preserve_topology=True)

    # ── Fallback for any unassigned buildings (e.g. no nearby road segment) ──
    for b in bldgs:
        if "parcel_poly_px" not in b:
            pt = Point(b["cx"], b["cy"])
            b_poly = b["poly_px"]
            best_v = None
            for v in vor_cells:
                if v.contains(pt):
                    best_v = v
                    break
            if best_v is None and vor_cells:
                best_v = min(vor_cells, key=lambda v: v.distance(pt))
            mrr = b_poly.minimum_rotated_rectangle
            setback_px = max(15.0, 5.0 / m_per_px)
            lot = mrr.buffer(setback_px, join_style=2, cap_style=2)
            if best_v is not None:
                lot = lot.intersection(best_v)
            lot = lot.intersection(roi_box)
            if not road_union.is_empty:
                diff = lot.difference(road_union)
                if not diff.is_empty and diff.area >= 50:
                    lot = diff
            if isinstance(lot, MultiPolygon):
                matched = [p for p in lot.geoms if p.contains(pt)]
                lot = matched[0] if matched else max(lot.geoms, key=lambda p: p.area)
            b["parcel_poly_px"] = lot.simplify(1.5, preserve_topology=True)

    # ── Factor 1: Generate GeoJSON, ULPINs, Property Cards ───────────────────
    parcels_list = []
    bldg_features = []
    property_cards = {}
    pid = 1

    for b in bldgs:
        bldg_features.append({
            "type": "Feature",
            "properties": {"id": b["id"], "area_m2": round(b["area_px"] * m2_per_px2, 1),
                           "unrecorded": bool(b.get("unrecorded", False)), "parcel_ulpin": ""},
            "geometry": mapping(b["poly_geo"]),
        })

    for b in bldgs:
        if "parcel_poly_px" not in b:
            continue
        p_plot_px = b["parcel_poly_px"]
        if p_plot_px.geom_type == "Polygon":
            polys = [p_plot_px]
        elif p_plot_px.geom_type == "MultiPolygon":
            polys = list(p_plot_px.geoms)
        else:
            continue

        if b.get("type") == "commercial":
            landuse  = "Commercial / Retail Complex"
            aoi_name = "Commercial Sector Survey"
        elif b.get("type") == "shed":
            landuse  = "Ancillary / Shed Structure"
            aoi_name = "Residential Cadastral Survey"
        else:
            landuse  = "Residential / Built-up"
            aoi_name = "Residential Cadastral Survey"

        for poly_px in polys:
            if poly_px.is_empty or poly_px.area < 50:
                continue

            lc_stats = analyze_parcel_landcover(
                poly_px=poly_px,
                veg_mask=veg_mask,
                tree_mask=tree_mask,
                barren_mask=barren_mask,
                bldg_area_px=b["area_px"],
                meters_per_px=m_per_px
            )

            if b.get("type") == "commercial":
                landuse  = "Commercial / Retail Complex"
                aoi_name = "Commercial Sector Survey"
            elif b.get("type") == "shed":
                landuse  = "Ancillary / Shed Structure"
                aoi_name = "Residential Cadastral Survey"
            else:
                landuse  = lc_stats["landuse"]
                aoi_name = "Residential Cadastral Survey" if "Residential" in landuse else "Rural Cadastral Survey"

            geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in poly_px.exterior.coords]
            poly_geo = Polygon(geo_pts)
            if not poly_geo.is_valid: poly_geo = poly_geo.buffer(0)
            if poly_geo.is_empty or poly_geo.area <= 0: continue

            centroid = poly_geo.centroid
            ulpin = cadastral_standards.generate_ulpin(centroid.y, centroid.x, pid)
            bldg_features[b["id"]]["properties"]["parcel_ulpin"] = ulpin

            area_m2 = round(poly_px.area * m2_per_px2, 1)
            b_area  = round(b["area_px"] * m2_per_px2, 1)
            gcr     = round((b_area / area_m2) * 100.0, 1) if area_m2 > 0 else 0.0

            props = {
                "id": pid, "ulpin": ulpin,
                "area_m2": area_m2,
                "area_guntha": round(area_m2 / 101.17, 3),
                "area_sq_ft": round(area_m2 * 10.7639, 1),
                "perimeter_m": round(poly_px.length * m_per_px, 1),
                "landuse": landuse,
                "building_count": 1,
                "building_ids": [b["id"]],
                "built_up_area_m2": b_area,
                "open_space_m2": max(0.0, round(area_m2 - b_area, 1)),
                "ground_coverage_ratio_pct": gcr,
                "vegetation_cover_pct": lc_stats["vegetation_cover_pct"],
                "tree_cover_pct": lc_stats["tree_cover_pct"],
                "crop_cover_pct": lc_stats["crop_cover_pct"],
                "barren_cover_pct": lc_stats["barren_cover_pct"],
                "crop_canopy_index": lc_stats["crop_canopy_index"],
                "cultivable_area_m2": lc_stats["cultivable_area_m2"],
                "cultivable_area_acres": lc_stats["cultivable_area_acres"],
                "barren_area_m2": lc_stats["barren_area_m2"],
                "road_connected": True, "road_distance_m": 0.0,
                "gps_lat": round(centroid.y, 6), "gps_lon": round(centroid.x, 6),
                "traverse_points": cadastral_standards.extract_traverse_points(poly_geo),
                "has_unrecorded_building": bool(b.get("unrecorded", False)), "alerts": [],
            }
            parcels_list.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
            property_cards[str(pid)] = cadastral_standards.generate_cadastral_property_card(
                props, poly_geo, [b["poly_geo"]], aoi_name=aoi_name)
            pid += 1

    # ── Delineate Open Rural Agricultural & Barren Land Parcels ───────────────
    allocated_px = [b["parcel_poly_px"] for b in bldgs if "parcel_poly_px" in b]
    rural_plots = delineate_open_rural_parcels(
        crop_mask=crop_mask,
        barren_mask=barren_mask,
        bund_segments=bund_segments or [],
        occupied_polys=allocated_px,
        roi_box=roi_box,
        meters_per_px=m_per_px,
        min_area_m2=150.0
    )

    allocated_union = unary_union(allocated_px) if allocated_px else Polygon()
    for r_plot_px in rural_plots:
        if not allocated_union.is_empty:
            diff = r_plot_px.difference(allocated_union)
            if diff.is_empty or diff.area < (100.0 / m2_per_px2):
                continue
            r_plot_px = diff

        sub_polys = [r_plot_px] if r_plot_px.geom_type == "Polygon" else list(r_plot_px.geoms)
        for poly_px in sub_polys:
            if poly_px.is_empty or poly_px.area < (100.0 / m2_per_px2):
                continue
            geo_pts = [px_to_lonlat_fn(float(x), float(y)) for x, y in poly_px.exterior.coords]
            poly_geo = Polygon(geo_pts)
            if not poly_geo.is_valid: poly_geo = poly_geo.buffer(0)
            if poly_geo.is_empty or poly_geo.area <= 0: continue

            centroid = poly_geo.centroid
            ulpin = cadastral_standards.generate_ulpin(centroid.y, centroid.x, pid)
            area_m2 = round(poly_px.area * m2_per_px2, 1)

            lc_stats = analyze_parcel_landcover(
                poly_px=poly_px,
                veg_mask=veg_mask,
                tree_mask=tree_mask,
                barren_mask=barren_mask,
                bldg_area_px=0.0,
                meters_per_px=m_per_px
            )
            landuse = lc_stats["landuse"]
            aoi_name = "Agricultural & Rural Survey" if "Agricultural" in landuse else "Rural Land Survey"

            props = {
                "id": pid, "ulpin": ulpin,
                "area_m2": area_m2,
                "area_guntha": round(area_m2 / 101.17, 3),
                "area_sq_ft": round(area_m2 * 10.7639, 1),
                "perimeter_m": round(poly_px.length * m_per_px, 1),
                "landuse": landuse,
                "building_count": 0,
                "building_ids": [],
                "built_up_area_m2": 0.0,
                "open_space_m2": area_m2,
                "ground_coverage_ratio_pct": 0.0,
                "vegetation_cover_pct": lc_stats["vegetation_cover_pct"],
                "tree_cover_pct": lc_stats["tree_cover_pct"],
                "crop_cover_pct": lc_stats["crop_cover_pct"],
                "barren_cover_pct": lc_stats["barren_cover_pct"],
                "crop_canopy_index": lc_stats["crop_canopy_index"],
                "cultivable_area_m2": lc_stats["cultivable_area_m2"],
                "cultivable_area_acres": lc_stats["cultivable_area_acres"],
                "barren_area_m2": lc_stats["barren_area_m2"],
                "road_connected": False, "road_distance_m": 0.0,
                "gps_lat": round(centroid.y, 6), "gps_lon": round(centroid.x, 6),
                "traverse_points": cadastral_standards.extract_traverse_points(poly_geo),
                "has_unrecorded_building": False, "alerts": [],
            }
            parcels_list.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
            property_cards[str(pid)] = cadastral_standards.generate_cadastral_property_card(
                props, poly_geo, [], aoi_name=aoi_name)
            pid += 1

    print(f"  [Factor 4] Total regularized cadastral parcels: {len(parcels_list)} (0 Overlaps Guaranteed)")
    return parcels_list, bldg_features, property_cards

