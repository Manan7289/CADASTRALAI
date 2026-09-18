"""
AI-Based Automated Urban Parcel Mapping — Rectilinear Subdivision Cadastral Engine
===================================================================================
Generates 100% distinct, street-aligned rectangular & trapezoidal cadastral land parcels.

Key Features:
  1. Guaranteed 1-to-1 Building-to-Parcel Subdivision (No multi-structure merged lots)
  2. Side lot lines run perpendicular to fronting streets
  3. Front lot lines align with street right-of-ways
  4. Unified commercial compound lots for retail complexes
  5. Cadastral Engine -> Assigns 14-digit ULPINs, property cards, and DXF exports
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, box, mapping
from shapely.ops import unary_union

backend_dir = Path(__file__).resolve().parent
BASE_DIR = backend_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir / "ml"))

import cadastral_standards
from building_classifier import classify_buildings, train as train_bldg
from boundary_detector import detect_physical_walls_and_fences, snap_polygon_to_physical_walls
from quad_regularizer import regularize_to_quadrilateral, normalize_block_parcel_areas
from parcel_engine import generate_4factor_cadastral_parcels

OUT_DIR = BASE_DIR / "data" / "uploads" / "clear_drone_survey"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load orthomosaic + GT mask ────────────────────────────────────────────
src_img_path = BASE_DIR / "data" / "processed" / "austin_sample.jpg"
img_pil = Image.open(src_img_path)
img_pil.save(OUT_DIR / "image.png")
img_bgr = cv2.imread(str(src_img_path))
H, W = img_bgr.shape[:2]  # 2000 x 2000

gt_path  = BASE_DIR / "data" / "datasets" / "inria_raw" / "data" / "train" / "gt" / "austin1.tif"
if not gt_path.exists():
    gt_path = Path("D:/cadastraai_data/inria_raw/data/train/gt/austin1.tif")
gt_raw   = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
crop_gt  = gt_raw[500:2500, 500:2500] if gt_raw is not None else None

# ── 2. WGS84 georef — Austin TX ───────────────────────────────────────────────
CROP_UL_LAT =  30.229612
CROP_UL_LON = -97.787781
CROP_LR_LAT =  30.224141
CROP_LR_LON = -97.781613

center_lat = (CROP_UL_LAT + CROP_LR_LAT) / 2.0
center_lon = (CROP_UL_LON + CROP_LR_LON) / 2.0

geo = {
    "width": W, "height": H,
    "lon_nw": CROP_UL_LON, "lat_nw": CROP_UL_LAT,
    "lon_se": CROP_LR_LON, "lat_se": CROP_LR_LAT,
    "source": "High-Resolution UAV Drone Orthomosaic (0.3m GSD) — Inria Dataset",
}

def px_to_lonlat(px, py):
    lon = geo["lon_nw"] + (px / W) * (geo["lon_se"] - geo["lon_nw"])
    lat = geo["lat_nw"] - (py / H) * (geo["lat_nw"] - geo["lat_se"])
    return round(lon, 7), round(lat, 7)

def latlon_to_px(lat, lon):
    px = (lon - geo["lon_nw"]) / (geo["lon_se"] - geo["lon_nw"]) * W
    py = (geo["lat_nw"] - lat) / (geo["lat_nw"] - geo["lat_se"]) * H
    return px, py

# ── 3. Parse OSM road network & build road corridors ──────────────────────────
print("=" * 60)
print("STEP 1: Parse Road Network & Land Sectors")
print("=" * 60)

osm_path = BASE_DIR / "data" / "raw" / "osm_bbox.json"
if not osm_path.exists():
    osm_path = Path("D:/cadastraai_data/raw_parcels/osm_bbox.json")
osm_data = json.loads(osm_path.read_text(encoding="utf-8"))
elements = osm_data.get("elements", [])
nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e["type"] == "node" and "lat" in e}
ways  = [e for e in elements if e["type"] == "way"]

ROAD_W = {
    "motorway": 22, "trunk": 18, "primary": 14, "secondary": 12,
    "tertiary": 10, "unclassified": 8, "residential": 7, "service": 5,
}

road_lines = []
road_polys = []
street_segs = []

for w in ways:
    tags = w.get("tags", {})
    if "highway" not in tags: continue
    hw = tags["highway"]
    width_px = ROAD_W.get(hw, 6)
    refs = w.get("nodes", [])
    pts = [latlon_to_px(*nodes[r]) for r in refs if r in nodes]
    if len(pts) >= 2:
        ls = LineString(pts)
        if ls.length > 0:
            road_lines.append((ls, width_px))
            road_polys.append(ls.buffer(width_px / 2.0, cap_style=2, join_style=2))
            p1, p2 = pts[0], pts[-1]
            ang = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])) % 180
            street_segs.append((ang, (p1, p2)))

road_union = unary_union(road_polys) if road_polys else Polygon()
roi_box    = box(0, 0, W, H)

print(f"  Rasterized {len(road_polys)} road corridors")

# ── 4. Extract building contours ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building Extraction & ML Classification")
print("=" * 60)
train_bldg()

contours, _ = cv2.findContours(crop_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bldgs = []
bldg_features = []
bldg_contours = []

for c in contours:
    a = cv2.contourArea(c)
    if a >= 70:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            eps = 0.015 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(approx) >= 3:
                geo_ring = [px_to_lonlat(float(x), float(y)) for x, y in approx]
                if geo_ring[0] != geo_ring[-1]:
                    geo_ring.append(geo_ring[0])
                poly_geo = Polygon(geo_ring)
                bldg_poly_px = Polygon(approx)
                if poly_geo.is_valid and poly_geo.area > 0:
                    bldgs.append({
                        "id": len(bldgs),
                        "cx": cx, "cy": cy,
                        "contour": c,
                        "approx": approx,
                        "poly_px": bldg_poly_px,
                        "poly_geo": poly_geo,
                        "area_px": a,
                    })
                    bldg_contours.append(c)

print(f"  Extracted {len(bldgs)} building plinths")

# Classify buildings with ML
bldg_types = classify_buildings(bldg_contours)
for b, t in zip(bldgs, bldg_types):
    b["type"] = t

for b in bldgs:
    bldg_features.append({
        "type": "Feature",
        "properties": {"id": b["id"], "area_m2": round(b["area_px"] * 0.09, 1),
                       "unrecorded": False, "parcel_ulpin": ""},
        "geometry": mapping(b["poly_geo"]),
    })

comm_bldgs = [b for b in bldgs if b["type"] == "commercial"]
res_bldgs  = [b for b in bldgs if b["type"] != "commercial"]
print(f"  ML Classification: {len(comm_bldgs)} commercial, {len(res_bldgs)} residential/shed")

# ── 5. 4-Factor Cadastral Parcel Engine ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: 4-Factor Cadastral Engine (Physical Walls + Quad + Equal-Area + OSM Topology)")
print("=" * 60)

parcels_list, bldg_features, property_cards = generate_4factor_cadastral_parcels(
    bldgs=bldgs,
    road_lines=road_lines,
    img_bgr=img_bgr,
    roi_box=roi_box,
    px_to_lonlat_fn=px_to_lonlat,
    cadastral_standards=cadastral_standards,
    road_union=road_union,
)

# ── 7. Road centerlines layer ────────────────────────────────────────────────
road_lines_geo = []
for i, (ang, (pt1, pt2)) in enumerate(street_segs[:50]):
    road_lines_geo.append({
        "type": "Feature",
        "properties": {"id": i, "name": f"Detected Road {i+1}", "angle_deg": round(ang, 1)},
        "geometry": {"type": "LineString",
                     "coordinates": [list(px_to_lonlat(*pt1)), list(px_to_lonlat(*pt2))]},
    })

# ── 8. Save all session files ────────────────────────────────────────────────
roads_fc     = {"type": "FeatureCollection", "features": road_lines_geo}
parcels_fc   = {"type": "FeatureCollection", "features": parcels_list}
buildings_fc = {"type": "FeatureCollection", "features": bldg_features}

(OUT_DIR / "meta.json").write_text(json.dumps({
    "meta": {
        "area":           f"Austin TX Urban Cadastral Survey ({center_lat:.4f}N, {abs(center_lon):.4f}W)",
        "source":         "UAV Orthomosaic 0.3m/px — Inria Aerial Image Dataset",
        "methods":        ["Rectilinear Subdivision Engine", "Street-Aligned Orthogonal Regularization", "KMeans Building Classification"],
        "gsd_m":          0.3,
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
    "session": "clear_drone_survey",
    "layers": {
        "parcels":   "parcels.geojson",
        "buildings": "buildings.geojson",
        "roads":     "roads.geojson",
        "dxf":       "cadastral_survey.dxf",
    }
}, indent=2))

print("\n" + "=" * 60)
print("COMPLETE — Rectilinear Cadastral Subdivision")
print(f"  Parcels  : {len(parcels_list)}")
print(f"  Buildings: {len(bldg_features)}")
print(f"  Output   : {OUT_DIR}")
print("=" * 60)
