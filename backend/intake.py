"""New-survey intake: store uploaded rasters, preview them, and run the
AI pipeline as a background job with progress the page can poll.

Accepted inputs
  ORI (required)   GeoTIFF orthoimage. A plain photo (JPG/PNG) or a short
                   video (its middle frame) is also accepted if the user gives
                   a centre point and ground width -- it is then written out as
                   a georeferenced GeoTIFF, so the rest of the pipeline is identical.
  DSM (optional)   GeoTIFF surface model. With a DTM -> nDSM = DSM - DTM;
                   without one, the DTM is derived by dtm.py's ground filter.
  DTM (optional)   GeoTIFF terrain model.
"""
import json
import math
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT

import parcel_extract
import segment
import survey

BASE = Path(__file__).resolve().parent.parent / "data" / "intake"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
PLAIN_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_AOI_KM2 = 0.35          # keeps a run under a couple of minutes on a laptop
PREVIEW_MAX_PX = 1600

_jobs = {}
_lock = threading.Lock()
Image.MAX_IMAGE_PIXELS = None


def _upload_dir(uid):
    d = BASE / uid
    if not d.is_dir() or "/" in uid or ".." in uid:
        raise FileNotFoundError(uid)
    return d


def _read_frame(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read a frame from the uploaded video.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def georeference_plain(path, out_path, center_lat, center_lon, width_m):
    """Write a plain image (or a video's middle frame) as a north-up GeoTIFF
    from a centre point and ground width. Approximate by nature -- the survey
    metadata records that it was manually georeferenced."""
    if Path(path).suffix.lower() in VIDEO_EXTS:
        arr = _read_frame(path)
    else:
        arr = np.asarray(Image.open(path).convert("RGB"))
    h, w = arr.shape[:2]
    height_m = width_m * h / w
    dlon = (width_m / 2) / (111320.0 * math.cos(math.radians(center_lat)))
    dlat = (height_m / 2) / 111320.0
    transform = from_bounds(center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat, w, h)
    with rasterio.open(out_path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=transform, compress="deflate", tiled=True) as dst:
        dst.write(arr.transpose(2, 0, 1))
    return {"georeferencing": f"manual: centre {center_lat:.6f},{center_lon:.6f}, ground width {width_m:g} m"}


def describe_raster(path):
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{Path(path).name} has no coordinate reference system.")
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        crs_m = segment.utm_crs_for((w + e) / 2, (s + n) / 2)
        l, b, r, t = transform_bounds(src.crs, crs_m, *src.bounds)
        gsd = (r - l) / src.width
        return {"bounds": [w, s, e, n], "width_px": src.width, "height_px": src.height, "bands": src.count,
                "gsd_cm": round(gsd * 100, 2), "area_km2": round((r - l) * (t - b) / 1e6, 4), "crs": src.crs.to_string()}


def make_preview(path, out_png):
    """Small north-up EPSG:3857 preview of the whole raster (overviews make
    this fast even for huge mosaics) plus its lat/lon bounds for Leaflet."""
    with rasterio.open(path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        l, b, r, t = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        scale = max(r - l, t - b) / PREVIEW_MAX_PX
        pw, ph = max(1, int((r - l) / scale)), max(1, int((t - b) / scale))
        with WarpedVRT(src, crs="EPSG:3857", transform=from_bounds(l, b, r, t, pw, ph), width=pw, height=ph,
                       resampling=Resampling.average) as vrt:
            rgb = vrt.read([1, 2, 3], masked=True)
    alpha = np.where(np.ma.getmaskarray(rgb).any(0) | (rgb.filled(0).sum(0) == 0), 0, 255).astype(np.uint8)
    arr = np.dstack([rgb.filled(0).transpose(1, 2, 0).clip(0, 255).astype(np.uint8), alpha])
    Image.fromarray(arr, "RGBA").save(out_png, optimize=True)
    return [[s, w], [n, e]]


def create_upload(files, form):
    """files: dict role -> werkzeug FileStorage; form: request.form."""
    ori = files.get("ori")
    if ori is None or not ori.filename:
        raise ValueError("An orthoimage (ORI) is required.")
    uid = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]
    d = BASE / uid
    d.mkdir(parents=True, exist_ok=True)
    info = {"id": uid, "files": {}, "notes": []}

    ext = Path(ori.filename).suffix.lower()
    if ext in PLAIN_IMAGE_EXTS | VIDEO_EXTS:
        raw = d / ("ori_raw" + ext)
        ori.save(raw)
        try:
            lat, lon, width_m = float(form["center_lat"]), float(form["center_lon"]), float(form["width_m"])
        except (KeyError, ValueError):
            raise ValueError("This image has no georeferencing: give a centre latitude, longitude and ground width in metres.")
        if width_m <= 0:
            raise ValueError("Ground width must be positive.")
        info["notes"].append(georeference_plain(raw, d / "ori.tif", lat, lon, width_m)["georeferencing"])
        if ext in VIDEO_EXTS:
            info["notes"].append("video: middle frame used (single frame, not a stitched mosaic)")
    else:
        ori.save(d / "ori.tif")
    info["files"]["ori"] = "ori.tif"

    for role in ("dsm", "dtm"):
        f = files.get(role)
        if f is not None and f.filename:
            f.save(d / f"{role}.tif")
            describe_raster(d / f"{role}.tif")   # fail fast on a DSM/DTM without a CRS
            info["files"][role] = f"{role}.tif"

    info["ori"] = describe_raster(d / "ori.tif")
    info["preview_bounds"] = make_preview(d / "ori.tif", d / "preview.png")
    info["max_aoi_km2"] = MAX_AOI_KM2
    (d / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def aoi_area_km2(aoi):
    w, s, e, n = aoi
    return (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * (n - s) * 111.32


def _set(job_id, **kw):
    with _lock:
        _jobs[job_id].update(kw)


def get_job(job_id):
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


KAGGLE_STAGES = ["Reading imagery and height data", "Uploading the image to Kaggle",
                 "Running the approved models on Kaggle GPU", "Downloading the results",
                 "Building buildings, roads and parcels; validating topology"]
STAGES = ["Reading imagery and height data", "Running AI segmentation", "Extracting buildings, corridors and parcels",
          "Validating topology and saving survey"]


def start_job(uid, name, aoi, model="auto"):
    d = _upload_dir(uid)
    info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    if aoi is None:
        aoi = info["ori"]["bounds"]
    if aoi_area_km2(aoi) > MAX_AOI_KM2:
        raise ValueError(f"Selected area is {aoi_area_km2(aoi):.2f} km²; draw an area of at most {MAX_AOI_KM2} km².")
    model_key, model_path, model_info = segment.choose_model(model, "dsm" in info["files"])
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        _jobs[job_id] = {"id": job_id, "state": "running", "stage": 0,
                         "stages": KAGGLE_STAGES if model_info.get("kind") == "kaggle" else STAGES, "started": time.time(),
                         "model": model_info["label"]}
    threading.Thread(target=_run, args=(job_id, d, info, name, aoi, model_key, model_path, model_info), daemon=True).start()
    return job_id


def _run(job_id, d, info, name, aoi, model_key, model_path, model_info):
    try:
        files = info["files"]
        _set(job_id, stage=0)
        gsd = model_info.get("gsd_m", 0.10)
        loaded = segment.load_survey(d / files["ori"], dsm_path=d / files["dsm"] if "dsm" in files else None,
                                     dtm_path=d / files["dtm"] if "dtm" in files else None, aoi_lonlat=aoi, gsd_m=gsd)
        _set(job_id, stage=1)
        notes = list(info.get("notes", []))
        if loaded["height_source"]:
            notes.append(loaded["height_source"])
        source = f"Uploaded ORI ({info['ori']['gsd_cm']} cm native, processed at {round(gsd * 100)} cm)" + (" · " + "; ".join(notes) if notes else "")
        model_meta = {"key": model_key, "name": model_info["label"], "summary": model_info["summary"], "limits": model_info["limits"]}
        if model_info.get("kind") == "kaggle":
            import import_bundle
            import kaggle_jobs
            job = kaggle_jobs.new_job_dir(job_id)
            kaggle_jobs.prepare(job, "upload", loaded, name or "Untitled survey", f"Uploaded ORI ({info['ori']['gsd_cm']} cm native)")
            names = KAGGLE_STAGES
            bundles = kaggle_jobs.run(job, stage=lambda s: _set(job_id, stage=names.index(s) if s in names else 1))
            if not bundles:
                raise RuntimeError("The Kaggle run finished without a result.")
            _set(job_id, stage=len(names) - 1)
            meta = import_bundle.import_bundle(bundles[0], loaded=loaded, name=name or "Untitled survey",
                                               source=source + " · models run on Kaggle GPU")
            _set(job_id, state="done", survey_id=meta["id"], seconds=round(time.time() - get_job(job_id)["started"], 1))
            return
        if model_info.get("kind") == "landcover":
            import import_bundle
            lcp = segment.predict_landcover(loaded["rgb"], model_path)
            _set(job_id, stage=2)
            meta = import_bundle.build_survey(name or "Untitled survey", source, loaded, lcp, model_meta)
            _set(job_id, state="done", survey_id=meta["id"], seconds=round(time.time() - get_job(job_id)["started"], 1))
            return
        probs, _ = segment.predict(loaded["rgb"], loaded["ndsm"], model_path=model_path)
        _set(job_id, stage=2)
        extracted = parcel_extract.extract(probs, loaded["rgb"], loaded["transform"], ndsm=loaded["ndsm"], valid=loaded["valid"])
        _set(job_id, stage=3)
        meta = survey.create(name or "Untitled survey", source, loaded, probs, extracted, model_meta)
        _set(job_id, state="done", survey_id=meta["id"], seconds=round(time.time() - get_job(job_id)["started"], 1))
    except Exception as e:
        traceback.print_exc()
        _set(job_id, state="error", error=str(e))
