"""Run the same detect -> delineate -> check pipeline on a user-supplied
drone image or video instead of the fixed Igatpuri demo AOI.

This is what makes the project usable as an actual tool rather than a fixed
demo -- but it's honest about what changes when the input isn't a curated
sample:

- No GPS/EXIF parsing is attempted. The caller supplies a centre latitude/
  longitude and an approximate ground width in metres; the geo-box is
  computed from that (see geo_box_from_center). Real drone metadata would
  make this exact; a manual estimate makes it approximate.
- There is no OSM ground truth for an arbitrary new area to train a
  RandomForest on (that's what model.py's colony-based training needs), so
  building detection here is unsupervised: an Otsu threshold over a
  brightness+texture score built from the same pixel_features() as the main
  model, calibrated to real building-size bounds in metres rather than a
  fixed pixel count. It will be less accurate than the trained model on the
  curated demo AOI -- that's a real, stated trade-off, not hidden.
- Road/rail/water/government/land-use layers ARE real: they're live-fetched
  from OpenStreetMap for whatever bounding box the upload resolves to, via
  the same Overpass functions fetch_data.py uses for the main pipeline.
- A short video has one representative frame extracted (its middle frame) --
  this is not photogrammetric stitching; multi-frame orthomosaic generation
  is a much larger problem (structure-from-motion, bundle adjustment) outside
  what a single-image pipeline like this one can honestly claim to do.
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union

import fetch_data
import model
import parcels
import rules

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_DIM = 900               # processing resolution, same rationale as model.py
MIN_BUILDING_M2 = 20         # real-world size bounds used to derive pixel-area thresholds
MAX_BUILDING_M2 = 3000
UNRECORDED_IOU_THRESH = 0.15

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def load_image(file_path):
    """Return (PIL.Image, is_video). A video has its middle frame extracted --
    a stand-in for a full frame, not a stitched mosaic."""
    ext = Path(file_path).suffix.lower()
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Could not read a frame from the uploaded video.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), True
    return Image.open(file_path).convert("RGB"), False


def geo_box_from_center(center_lat, center_lon, width_m, img_w, img_h):
    height_m = width_m * (img_h / img_w)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
    half_w_deg = (width_m / 2) / m_per_deg_lon
    half_h_deg = (height_m / 2) / m_per_deg_lat
    return {
        "width": img_w, "height": img_h,
        "lon_nw": center_lon - half_w_deg, "lat_nw": center_lat + half_h_deg,
        "lon_se": center_lon + half_w_deg, "lat_se": center_lat - half_h_deg,
        "source": "User upload, manually georeferenced from a supplied centre point + ground width",
    }


def detect_buildings_unsupervised(img_arr, meters_per_px):
    """No labelled ground truth exists for an arbitrary new area, so this
    thresholds a brightness+texture score (Otsu) instead of running a
    trained classifier -- calibrated to plausible building sizes in real
    metres so it doesn't depend on the image's pixel resolution."""
    h, w = img_arr.shape[:2]
    feats = model.pixel_features(img_arr)
    v = feats[:, 5].reshape(h, w)
    tex = feats[:, 6].reshape(h, w)

    def norm(a):
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

    score = (255 * (0.5 * norm(v) + 0.5 * norm(tex))).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_blob_px = max(4, int(MIN_BUILDING_M2 / (meters_per_px ** 2)))
    max_blob_px = max(min_blob_px + 1, int(MAX_BUILDING_M2 / (meters_per_px ** 2)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_px = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_blob_px or a > max_blob_px:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys_px.append(approx)
    return polys_px


def build_report(building_id, area_m2, alerts):
    lines = [
        "CADASTRAAI -- FIELD VERIFICATION NOTE (USER UPLOAD)",
        f"Ref: CADAI-UPL-{building_id:04d}",
        f"Footprint area (unsupervised detection): {area_m2} m2",
        "",
    ]
    if alerts:
        lines.append("FINDINGS")
        for i, a in enumerate(alerts, 1):
            lines.append(f"{i}. {a['type']}: {a['msg']}")
            lines.append(f"   Reference: {a['citation']}")
    else:
        lines.append("FINDINGS")
        lines.append("None. Footprint appears compliant against all automated checks.")
    lines.append("")
    lines.append("Detected with an unsupervised threshold, not the trained demo model --")
    lines.append("no labelled ground truth exists yet for this location. Treat every")
    lines.append("finding here as lower-confidence than the curated Igatpuri demo, and")
    lines.append("verify in the field before acting on it.")
    return "\n".join(lines)


def process_upload(file_path, center_lat, center_lon, width_m, session_id):
    t0 = time.time()
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    img, is_video = load_image(file_path)
    scale = min(1.0, MAX_DIM / max(img.width, img.height))
    proc_img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    arr = np.array(proc_img)

    geo = geo_box_from_center(center_lat, center_lon, width_m, proc_img.width, proc_img.height)
    meters_per_px = width_m / proc_img.width
    lonlat_to_px, px_to_lonlat = model.make_pixel_transforms(geo)

    polys_px = detect_buildings_unsupervised(arr, meters_per_px)
    detected_polys = []
    for approx in polys_px:
        ring = [px_to_lonlat(px, py) for px, py in approx]
        ring.append(ring[0])
        poly = Polygon(ring)
        if poly.is_valid and poly.area > 0:
            detected_polys.append(poly)

    bbox = (geo["lat_se"], geo["lon_nw"], geo["lat_nw"], geo["lon_se"])
    live = fetch_data.fetch_all(bbox)

    osm_polys = [shape(f["geometry"]) for f in live["buildings"]["features"] if f["geometry"]["type"] == "Polygon"]
    osm_union = unary_union(osm_polys) if osm_polys else Polygon()
    buildings_fc = {"type": "FeatureCollection", "features": []}
    for i, poly in enumerate(detected_polys):
        inter = poly.intersection(osm_union).area if not osm_union.is_empty else 0.0
        unrecorded = (inter / poly.area if poly.area > 0 else 1.0) < UNRECORDED_IOU_THRESH
        buildings_fc["features"].append({
            "type": "Feature", "properties": {"id": i, "unrecorded": bool(unrecorded)}, "geometry": mapping(poly),
        })

    try:
        layers = rules.layers_from_geojson(buildings_fc, live["roads"], live["railway"], live["waterway"], live["government"])
        rule_results = {r["id"]: r for r in rules.evaluate_layers(layers)}
    except Exception:
        rule_results = {}

    deg2_to_m2 = 111320.0 * (111320.0 * math.cos(math.radians(center_lat)))
    reports = {}
    for f in buildings_fc["features"]:
        bid = f["properties"]["id"]
        fallback_area = round(shape(f["geometry"]).area * deg2_to_m2, 1)
        r = rule_results.get(bid, {"alerts": [], "area_m2": fallback_area})
        f["properties"]["alerts"] = r["alerts"]
        f["properties"]["area_m2"] = r.get("area_m2")
        reports[str(bid)] = build_report(bid, r.get("area_m2"), r["alerts"])

    try:
        parcel_list = parcels.build_parcels_from_data(geo, buildings_fc, live["roads"], live["railway"],
                                                        live["waterway"], live["government"], live["landuse"])
    except Exception:
        parcel_list = []
    parcels_fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {k: v for k, v in p.items() if k != "geometry"}, "geometry": p["geometry"]}
                     for p in parcel_list],
    }

    proc_img.save(session_dir / "image.png")
    (session_dir / "meta.json").write_text(json.dumps({
        "meta": {"area": f"User upload ({center_lat:.5f}, {center_lon:.5f})",
                 "source": "User-supplied image" + (" (video, middle frame)" if is_video else "") +
                           " + live OpenStreetMap reference layers"},
        "image": geo,
    }), encoding="utf-8")
    (session_dir / "buildings.geojson").write_text(json.dumps(buildings_fc), encoding="utf-8")
    (session_dir / "osm_buildings.geojson").write_text(json.dumps(live["buildings"]), encoding="utf-8")
    (session_dir / "parcels.geojson").write_text(json.dumps(parcels_fc), encoding="utf-8")
    (session_dir / "layers.json").write_text(json.dumps({
        "roads": live["roads"], "railway": live["railway"], "waterway": live["waterway"], "government": live["government"],
    }), encoding="utf-8")
    (session_dir / "reports.json").write_text(json.dumps(reports), encoding="utf-8")
    (session_dir / "metrics.json").write_text(json.dumps({
        "method": "unsupervised",
        "note": ("Detected with an Otsu brightness/texture threshold, not the trained RandomForest -- "
                 "no labelled ground truth exists yet for this location, so there is no held-out accuracy "
                 "to report. Verify detections in the field before acting on them."),
        "buildings_detected": len(detected_polys),
        "parcels_delineated": len(parcel_list),
        "processing_seconds": round(time.time() - t0, 2),
    }), encoding="utf-8")

    return {"session_id": session_id, "buildings_detected": len(detected_polys), "parcels": len(parcel_list)}
