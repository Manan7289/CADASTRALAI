"""
AI-Based Automated Cadastral Mapping — Indian Land Survey Engine
===============================================================
Location: Ajit Singh Nagar, Vijayawada, Andhra Pradesh, India (16.525°N, 80.637°E)
Data Source: 10cm GSD UAV Drone Orthomosaic + Google Open Buildings Dataset
Standards: DILRMP / SVAMITVA / ULPIN (Bhu-Aadhaar) / ISO 19152 LADM

Generates:
  1. Full Block Cadastral Parcels (100% gapless contiguous coverage)
  2. 1-to-1 Building Enclosure
  3. 14-digit ULPIN (Bhu-Aadhaar) per parcel
  4. Metric, Guntha, Sq Yard (Gaj), and Sq Ft area measurements
  5. Official Cadastral Property Cards (RoR) & DXF Export
"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, box, mapping, shape
from shapely.ops import unary_union, voronoi_diagram as shapely_voronoi
from shapely.geometry import MultiPoint

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "ml"))

import cadastral_standards
from building_classifier import classify_buildings, train as train_bldg
from road_detector import detect_roads

OUT_DIR = backend_dir.parent / "data" / "uploads" / "indian_drone_survey"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Vijayawada 10cm UAV Orthomosaic ──────────────────────────────────
vj_tif_path = Path("D:/cadastraai_data/custom_datasets/datasets/vijayawada/aoi_singhnagar_10cm.tif")
print("=" * 65)
print("CADASTRAAI — INDIAN MUNICIPAL CADASTRAL SURVEY ENGINE")
print("AOI: Ajit Singh Nagar, Vijayawada, Andhra Pradesh, India")
print("=" * 65)

img_bgr = cv2.imread(str(vj_tif_path))
H, W = img_bgr.shape[:2]  # 2488 x 2488

# Save preview image for web viewer
cv2.imwrite(str(OUT_DIR / "image.png"), img_bgr)
print(f"Loaded Indian UAV Drone Orthomosaic: {W}x{H} px (10cm GSD)")

# ── 2. Georeferencing (WGS84) ────────────────────────────────────────────────
# GeoTIFF metadata from header:
UL_LON = 80.63651936798892
UL_LAT = 16.52677975249633
SCALE_DEG = 9.373306035630019e-07  # ~0.104 m/px
LR_LON = UL_LON + W * SCALE_DEG
LR_LAT = UL_LAT - H * SCALE_DEG

center_lat = (UL_LAT + LR_LAT) / 2.0
center_lon = (UL_LON + LR_LON) / 2.0

geo = {
    "width": W, "height": H,
    "lon_nw": UL_LON, "lat_nw": UL_LAT,
    "lon_se": LR_LON, "lat_se": LR_LAT,
    "source": "Ajit Singh Nagar UAV Drone Orthomosaic (0.10m GSD) — Vijayawada AP, India",
}

def px_to_lonlat(px, py):
    lon = geo["lon_nw"] + (px / W) * (geo["lon_se"] - geo["lon_nw"])
    lat = geo["lat_nw"] - (py / H) * (geo["lat_nw"] - geo["lat_se"])
    return round(lon, 7), round(lat, 7)

def latlon_to_px(lat, lon):
    px = (lon - geo["lon_nw"]) / (geo["lon_se"] - geo["lon_nw"]) * W
    py = (geo["lat_nw"] - lat) / (geo["lat_nw"] - geo["lat_se"]) * H
    return px, py

# ── 3. Run ML Road & Street Network Inference ───────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Run ML Road & Street Network Classifier (road_clf.pkl)")
print("=" * 60)

road_mask, freeway_poly, street_segs = detect_roads(img_bgr)

# Construct realistic street right-of-way corridors (buffered centerlines + freeway)
street_lines = []
for ang, ((x1, y1), (x2, y2)) in street_segs:
    street_lines.append(LineString([(x1, y1), (x2, y2)]).buffer(10))  # ~2m half-width = 4m street

road_corridors = unary_union(street_lines + ([freeway_poly] if freeway_poly.area > 500 else []))
roi_box = box(0, 0, W, H)
print(f"  ML Detected: {len(street_segs)} street centerlines, freeway area: {freeway_poly.area:.0f} px2")

# ── 4. Load Building Footprints (Google Open Buildings India) ────────────────
print("\n" + "=" * 60)
print("STEP 2: Load Building Footprints & ML Morphology Classifier")
print("=" * 60)
train_bldg()

google_geojson_path = Path("D:/cadastraai_data/custom_datasets/datasets/footprints/google_vijayawada_mosaic.geojson")
with open(google_geojson_path) as f:
    fc_google = json.load(f)

aoi_poly_geo = box(UL_LON, LR_LAT, LR_LON, UL_LAT)
bldgs = []
bldg_features = []
bldg_contours = []

for feat in fc_google["features"]:
    geom_geo = shape(feat["geometry"])
    if not aoi_poly_geo.intersects(geom_geo):
        continue
    clipped_geo = geom_geo.intersection(aoi_poly_geo)
    if clipped_geo.is_empty or clipped_geo.area <= 0:
        continue
    
    # Extract outer polygon
    if clipped_geo.geom_type == "MultiPolygon":
        clipped_geo = max(clipped_geo.geoms, key=lambda g: g.area)
    if clipped_geo.geom_type != "Polygon":
        continue
    
    # Convert polygon coords from Lat/Lon to pixel coordinates
    px_ring = [latlon_to_px(lat, lon) for lon, lat in clipped_geo.exterior.coords]
    poly_px = Polygon(px_ring)
    if not poly_px.is_valid or poly_px.area < 50:
        continue
    
    cx, cy = poly_px.centroid.x, poly_px.centroid.y
    cnt_pts = np.array(px_ring, dtype=np.int32).reshape((-1, 1, 2))
    
    bldg_id = len(bldgs)
    bldgs.append({
        "id": bldg_id,
        "cx": cx, "cy": cy,
        "poly_px": poly_px,
        "poly_geo": clipped_geo,
        "area_px": poly_px.area,
        "contour": cnt_pts,
    })
    bldg_contours.append(cnt_pts)

print(f"  Extracted {len(bldgs)} Indian building plinths from Google Open Buildings")

# ML Building Classifier (Residential vs Commercial vs Ancillary Shed)
bldg_types = classify_buildings(bldg_contours)
for b, t in zip(bldgs, bldg_types):
    b["type"] = t

for b in bldgs:
    bldg_features.append({
        "type": "Feature",
        "properties": {
            "id": b["id"],
            "area_m2": round(b["area_px"] * 0.0108, 1),
            "area_sq_yards": round((b["area_px"] * 0.0108) / 0.8361, 1),
            "type": b["type"],
            "unrecorded": False,
            "parcel_ulpin": ""
        },
        "geometry": mapping(b["poly_geo"]),
    })

# ── 5. Full-Block Contiguous Cadastral Partition ────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Full-Block Contiguous Indian Cadastral Partition")
print("=" * 60)

bldg_seeds = MultiPoint([Point(b["cx"], b["cy"]) for b in bldgs])
roi_expanded = roi_box.buffer(10)
vor_cells = shapely_voronoi(bldg_seeds, envelope=roi_expanded)
cell_list = list(vor_cells.geoms)

for b in bldgs:
    pt = Point(b["cx"], b["cy"])
    best_cell = None
    for cell in cell_list:
        if cell.contains(pt):
            best_cell = cell
            break
    if best_cell is None:
        best_cell = min(cell_list, key=lambda c: c.distance(pt))
        
    clipped = best_cell.intersection(roi_box)
    if not road_corridors.is_empty:
        diff = clipped.difference(road_corridors)
        if not diff.is_empty and diff.area >= 100:
            clipped = diff
            
    if clipped.is_empty:
        continue
    if isinstance(clipped, MultiPolygon):
        matched = [p for p in clipped.geoms if p.contains(pt)]
        clipped = matched[0] if matched else max(clipped.geoms, key=lambda p: p.area)
        
    simplified = clipped.simplify(3.0, preserve_topology=True)
    if not simplified.is_valid or simplified.area < 100:
        simplified = clipped
        
    b["parcel_poly_px"] = simplified

# ── 6. Build Indian GeoJSON Parcels & Official Property Cards ───────────────
parcels_list = []
property_cards = {}
pid = 1

def make_parcel(p_plot_px, b, landuse, aoi_name, bldg_poly_list):
    global pid
    if p_plot_px.geom_type == "Polygon":
        polys = [p_plot_px]
    elif p_plot_px.geom_type == "MultiPolygon":
        polys = list(p_plot_px.geoms)
    else:
        return

    for poly_px in polys:
        if poly_px.is_empty or poly_px.area < 100:
            continue
        geo_pts = [px_to_lonlat(float(x), float(y)) for x, y in poly_px.exterior.coords]
        poly_geo = Polygon(geo_pts)
        if not poly_geo.is_valid: poly_geo = poly_geo.buffer(0)
        if poly_geo.is_empty or poly_geo.area <= 0: continue

        centroid = poly_geo.centroid
        # Generate 14-digit Indian standard ULPIN (Bhu-Aadhaar)
        ulpin = cadastral_standards.generate_ulpin(centroid.y, centroid.x, pid)
        if b is not None:
            bldg_features[b["id"]]["properties"]["parcel_ulpin"] = ulpin

        area_m2 = round(poly_px.area * 0.0108, 1)  # 10cm GSD: (0.104m)^2 = 0.0108 m2/px
        b_area  = round(b["area_px"] * 0.0108, 1) if b else 0.0
        gcr     = round((b_area / area_m2) * 100.0, 1) if area_m2 > 0 and b else 0.0

        # Indian Land Measurements
        area_guntha   = round(area_m2 / 101.17, 3)  # 1 Guntha = 101.17 sq meters
        area_sq_yards = round(area_m2 / 0.836127, 1) # 1 Gaj / Sq Yard = 0.8361 m2
        area_sq_ft    = round(area_m2 * 10.7639, 1)

        props = {
            "id": pid,
            "ulpin": ulpin,
            "bhu_aadhaar": ulpin,
            "state": "Andhra Pradesh",
            "district": "NTR / Krishna",
            "mandal": "Vijayawada Urban",
            "village_ward": "Ajit Singh Nagar",
            "area_m2": area_m2,
            "area_guntha": area_guntha,
            "area_sq_yards": area_sq_yards,
            "area_sq_ft": area_sq_ft,
            "perimeter_m": round(poly_px.length * 0.104, 1),
            "landuse": landuse,
            "building_count": 1 if b else 0,
            "building_ids": [b["id"]] if b else [],
            "built_up_area_m2": b_area,
            "open_space_m2": max(0.0, round(area_m2 - b_area, 1)),
            "ground_coverage_ratio_pct": gcr,
            "road_connected": True,
            "road_distance_m": 0.0,
            "gps_lat": round(centroid.y, 6),
            "gps_lon": round(centroid.x, 6),
            "traverse_points": cadastral_standards.extract_traverse_points(poly_geo),
            "has_unrecorded_building": False,
            "alerts": [],
        }
        parcels_list.append({"type": "Feature", "properties": props, "geometry": mapping(poly_geo)})
        property_cards[str(pid)] = cadastral_standards.generate_cadastral_property_card(
            props, poly_geo, bldg_poly_list, aoi_name=aoi_name)
        pid += 1

for b in bldgs:
    if "parcel_poly_px" not in b:
        continue
    if b["type"] == "commercial":
        landuse = "Commercial / Retail Enterprise"
        aoi_name = "Ajit Singh Nagar Commercial Ward"
    elif b["type"] == "shed":
        landuse = "Ancillary / Shed / Outbuilding"
        aoi_name = "Ajit Singh Nagar Residential Ward"
    else:
        landuse = "Residential Property (Abadi)"
        aoi_name = "Ajit Singh Nagar Residential Ward"

    make_parcel(b["parcel_poly_px"], b, landuse, aoi_name, [b["poly_geo"]])

print(f"  Generated {len(parcels_list)} Indian cadastral parcels (ULPIN Bhu-Aadhaar assigned)")

# ── 7. Road Centerlines GeoJSON ──────────────────────────────────────────────
road_lines_geo = []
for i, (ang, (pt1, pt2)) in enumerate(street_segs[:80]):
    road_lines_geo.append({
        "type": "Feature",
        "properties": {"id": i, "name": f"Road Link {i+1}", "angle_deg": round(ang, 1)},
        "geometry": {
            "type": "LineString",
            "coordinates": [list(px_to_lonlat(*pt1)), list(px_to_lonlat(*pt2))]
        }
    })

# ── 8. Save All Session Files ────────────────────────────────────────────────
roads_fc     = {"type": "FeatureCollection", "features": road_lines_geo}
parcels_fc   = {"type": "FeatureCollection", "features": parcels_list}
buildings_fc = {"type": "FeatureCollection", "features": bldg_features}

(OUT_DIR / "meta.json").write_text(json.dumps({
    "meta": {
        "area":           f"Ajit Singh Nagar Cadastral Survey, Vijayawada, AP ({center_lat:.4f}N, {center_lon:.4f}E)",
        "source":         "UAV Drone Orthomosaic (0.10m GSD) — Vijayawada Municipal Corporation",
        "methods":        ["Full-Block Contiguous Partition", "ULPIN Bhu-Aadhaar Assignment", "SVAMITVA Standards"],
        "gsd_m":          0.104,
        "parcels_count":   len(parcels_list),
        "buildings_count": len(bldg_features),
    },
    "image": geo,
}, indent=2))

(OUT_DIR / "parcels.geojson").write_text(json.dumps(parcels_fc, indent=2))
(OUT_DIR / "buildings.geojson").write_text(json.dumps(buildings_fc, indent=2))
(OUT_DIR / "roads.geojson").write_text(json.dumps(roads_fc, indent=2))
(OUT_DIR / "property_cards.json").write_text(json.dumps(property_cards, indent=2))

# DXF Export
cadastral_standards.export_cadastral_dxf(parcels_fc, buildings_fc, OUT_DIR / "cadastral_survey.dxf")

# Save layers.json
(OUT_DIR / "layers.json").write_text(json.dumps({
    "session": "indian_drone_survey",
    "layers": {
        "parcels":   "parcels.geojson",
        "buildings": "buildings.geojson",
        "roads":     "roads.geojson",
        "dxf":       "cadastral_survey.dxf",
    }
}, indent=2))

print("\n" + "=" * 60)
print("COMPLETE — Indian Cadastral Survey (Ajit Singh Nagar, Vijayawada)")
print(f"  Parcels  : {len(parcels_list)}")
print(f"  Buildings: {len(bldg_features)}")
print(f"  Output   : {OUT_DIR}")
print("=" * 60)
