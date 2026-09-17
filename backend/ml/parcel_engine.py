"""
Data-driven Parcel Engine.

Uses detected road segments to:
1. Identify dominant street grid angle per local neighbourhood
2. Align each parcel rectangle to its nearest road segment
3. Enforce zero-overlap via claimed_area union
"""
import numpy as np, cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union


def dominant_angles_from_segs(street_segs, n_bins=36):
    """
    Find the 1-2 dominant street grid directions from Hough segments.
    Returns list of dominant angles in degrees.
    """
    if not street_segs:
        return [0.0]
    angles = np.array([a % 90 for a, _ in street_segs])
    # Histogram over [0,90)
    counts, edges = np.histogram(angles, bins=n_bins, range=(0, 90))
    # Peak detection: find local maxima
    peaks = []
    for i in range(n_bins):
        l = counts[(i-1) % n_bins]
        r = counts[(i+1) % n_bins]
        if counts[i] >= l and counts[i] >= r and counts[i] > 0:
            peaks.append((counts[i], (edges[i]+edges[i+1])/2))
    peaks.sort(reverse=True)
    return [p[1] for p in peaks[:2]] if peaks else [0.0]


def nearest_road_angle(cx, cy, street_segs, search_radius=250):
    """
    Find the angle of the road segment closest to point (cx, cy).
    Falls back to 0 if no segments within search_radius.
    """
    best_d2 = search_radius**2
    best_ang = None
    for ang, ((x1,y1),(x2,y2)) in street_segs:
        # Distance from point to segment midpoint
        mx, my = (x1+x2)/2, (y1+y2)/2
        d2 = (cx-mx)**2 + (cy-my)**2
        if d2 < best_d2:
            best_d2 = d2
            best_ang = ang
    return best_ang  # None if nothing nearby


def generate_parcels(residential_bldgs, street_segs, freeway_poly, c_poly_px,
                     px_to_lonlat_fn, cadastral_standards,
                     bldg_features, property_cards, pid_start=3):
    """
    Generate zero-overlap cadastral parcels for residential buildings.

    Parameters
    ----------
    residential_bldgs   : list of building dicts with 'contour','approx','area','id','poly'
    street_segs         : list of (angle_deg, (pt1, pt2)) from road_detector
    freeway_poly        : Shapely Polygon (pixel coords) for freeway ROW
    c_poly_px           : Shapely Polygon (pixel coords) for commercial compound
    px_to_lonlat_fn     : function(px, py) -> (lon, lat)
    cadastral_standards : module
    bldg_features       : list (mutated: sets parcel_ulpin)
    property_cards      : dict (mutated: adds cards)
    pid_start           : first parcel ID for residential

    Returns
    -------
    parcels_list, pid_end
    """
    # Global dominant angle as fallback
    global_dominant = dominant_angles_from_segs(street_segs)
    fallback_angle  = global_dominant[0] if global_dominant else 0.0
    print(f"[parcel_engine] Global dominant street angles: {[f'{a:.1f}' for a in global_dominant]}")

    parcels_list = []
    pid = pid_start
    claimed_area = freeway_poly.union(c_poly_px).buffer(0)

    for b in residential_bldgs:
        rect = cv2.minAreaRect(b["contour"])
        (cx, cy), (w, h), cv_angle = rect

        long_side  = max(w, h)
        short_side = min(w, h)

        # Choose alignment angle:
        # 1. Try nearest road segment within 250px
        # 2. Fall back to global dominant angle
        road_angle = nearest_road_angle(cx, cy, street_segs, search_radius=250)
        if road_angle is not None:
            # Snap to nearest dominant angle within 15 degrees
            snapped = min(global_dominant, key=lambda a: abs(a - (road_angle % 90)))
            if abs(snapped - (road_angle % 90)) < 20:
                use_angle = snapped
            else:
                use_angle = road_angle % 90
        else:
            use_angle = fallback_angle

        plot_w = max(short_side + 20.0, 48.0)
        plot_h = max(long_side  + 60.0, 80.0)

        box_pts = cv2.boxPoints(((cx, cy), (plot_w, plot_h), use_angle))
        p_plot  = Polygon(box_pts)
        if not p_plot.is_valid:
            p_plot = p_plot.buffer(0)

        if p_plot.intersects(claimed_area):
            p_plot = p_plot.difference(claimed_area.buffer(0.8))
            if p_plot.geom_type == "MultiPolygon":
                bcp = Polygon(b["approx"]).buffer(0).centroid
                cont = [g for g in p_plot.geoms if g.contains(bcp)]
                p_plot = cont[0] if cont else (
                    max(p_plot.geoms, key=lambda g: g.area) if p_plot.geoms else Polygon())
            if not isinstance(p_plot, Polygon):
                p_plot = Polygon()

        if p_plot.is_empty or p_plot.area < 160:
            continue

        claimed_area = claimed_area.union(p_plot).buffer(0)

        geo_pts  = [px_to_lonlat_fn(float(x), float(y)) for x, y in p_plot.exterior.coords]
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid or poly_geo.area <= 0:
            continue

        centroid = poly_geo.centroid
        ulpin    = cadastral_standards.generate_ulpin(centroid.y, centroid.x, pid)
        bldg_features[b["id"]]["properties"]["parcel_ulpin"] = ulpin

        area_m2 = round(p_plot.area * 0.09, 1)
        b_area  = round(b["area"] * 0.09, 1)

        props = {
            "id": pid, "ulpin": ulpin,
            "area_m2": area_m2,
            "area_guntha": round(area_m2 / 101.17, 3),
            "area_sq_ft": round(area_m2 * 10.7639, 1),
            "perimeter_m": round(p_plot.length * 0.3, 1),
            "landuse": "Residential / Built-up",
            "building_count": 1,
            "building_ids": [b["id"]],
            "built_up_area_m2": b_area,
            "open_space_m2": max(0.0, round(area_m2 - b_area, 1)),
            "ground_coverage_ratio_pct": round((b_area / area_m2) * 100.0, 1) if area_m2 else 0.0,
            "road_connected": True, "road_distance_m": 0.0,
            "gps_lat": round(centroid.y, 6), "gps_lon": round(centroid.x, 6),
            "traverse_points": cadastral_standards.extract_traverse_points(poly_geo),
            "has_unrecorded_building": False, "alerts": [],
        }
        parcels_list.append({"type": "Feature", "properties": props,
                              "geometry": __import__("shapely.geometry", fromlist=["mapping"]).mapping(poly_geo)})
        property_cards[str(pid)] = cadastral_standards.generate_cadastral_property_card(
            props, poly_geo, [b["poly"]], aoi_name="Residential Cadastral Survey")
        pid += 1

    print(f"[parcel_engine] Generated {len(parcels_list)} residential parcels, 0 overlaps enforced")
    return parcels_list, pid
