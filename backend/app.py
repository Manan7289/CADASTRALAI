"""Local dev server for CadastraAI's static frontend, plus the upload
pipeline (which needs an actual Python process to run OpenCV/shapely/OSM
fetches on demand -- something a static host like Vercel can't do).

The frontend fetches everything (buildings, layers, metrics, reports) from
plain files under frontend/data/ -- the same files Vercel serves statically
in production (see build_static.py) -- unless a `?session=<id>` query param
is present, in which case it fetches from /uploads/<id>/ instead, which this
app populates via POST /api/process. The upload feature only works when this
Flask process is actually running; it is not available on the static
Vercel deployment.

Run order (fixed demo AOI): fetch_data.py -> fetch_imagery.py -> model.py -> build_static.py -> app.py
Run order (your own upload): just app.py, then use the Upload page.
"""
import json
import traceback
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import intake
import survey
import upload_pipeline
from export import export as export_parcels

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
RAW_UPLOADS_DIR = BASE_DIR / "data" / "raw" / "uploads"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # orthomosaics and DSMs are large


@app.route("/api/process", methods=["POST"])
def api_process():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400
    try:
        center_lat = float(request.form["center_lat"])
        center_lon = float(request.form["center_lon"])
        width_m = float(request.form["width_m"])
    except (KeyError, ValueError):
        return jsonify({"error": "center_lat, center_lon and width_m must all be provided as numbers."}), 400
    if width_m <= 0:
        return jsonify({"error": "width_m must be positive."}), 400

    RAW_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    saved_path = RAW_UPLOADS_DIR / (session_id + "_" + secure_filename(file.filename))
    file.save(saved_path)

    try:
        result = upload_pipeline.process_upload(saved_path, center_lat, center_lon, width_m, session_id)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {e}"}), 500

    return jsonify(result)


@app.route("/uploads/<session_id>/<path:filename>")
def serve_upload(session_id, filename):
    return send_from_directory(UPLOADS_DIR / secure_filename(session_id), filename)


# ---------------------------------------------------------------- survey workbench API
@app.route("/api/surveys")
def api_surveys():
    return jsonify(survey.list_surveys())


@app.route("/api/surveys/<sid>/parcels", methods=["PUT"])
def api_save_parcels(sid):
    body = request.get_json(silent=True)
    if not body or body.get("type") != "FeatureCollection":
        return jsonify({"error": "Expected a GeoJSON FeatureCollection."}), 400
    try:
        parcels_fc, issues_fc = survey.save_parcels(sid, body, action=request.args.get("action", "edited in workbench"))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    return jsonify({"parcels": parcels_fc, "issues": issues_fc})


@app.route("/api/surveys/<sid>/autofix", methods=["POST"])
def api_autofix(sid):
    try:
        parcels_fc, issues_fc, log = survey.auto_fix(sid)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    return jsonify({"parcels": parcels_fc, "issues": issues_fc, "log": log})


@app.route("/api/surveys/<sid>/merge", methods=["POST"])
def api_merge(sid):
    ids = (request.get_json(silent=True) or {}).get("ids", [])
    try:
        parcels_fc, issues_fc = survey.merge(sid, ids)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"parcels": parcels_fc, "issues": issues_fc})


@app.route("/api/surveys/<sid>/export")
def api_export(sid):
    fmt = request.args.get("fmt", "gpkg")
    try:
        fc = json.loads((survey.survey_dir(sid) / "parcels.geojson").read_text(encoding="utf-8"))
        data, name, mime = export_parcels(fc, fmt)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return Response(data, mimetype=mime, headers={"Content-Disposition": f"attachment; filename={sid}_{name}"})


@app.route("/surveys/<sid>/<path:filename>")
def serve_survey(sid, filename):
    try:
        d = survey.survey_dir(sid)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    resp = send_from_directory(d, filename)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- new survey intake
@app.route("/api/intake", methods=["POST"])
def api_intake():
    try:
        info = intake.create_upload(request.files, request.form)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not read the upload: {e}"}), 400
    return jsonify(info)


@app.route("/api/intake/<uid>/process", methods=["POST"])
def api_intake_process(uid):
    body = request.get_json(silent=True) or {}
    aoi = body.get("aoi")
    try:
        if aoi is not None:
            aoi = [float(v) for v in aoi]
            if len(aoi) != 4:
                raise ValueError("aoi must be [west, south, east, north].")
        job_id = intake.start_job(uid, (body.get("name") or "").strip()[:120], aoi)
    except FileNotFoundError:
        return jsonify({"error": "Unknown upload."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = intake.get_job(job_id)
    return (jsonify(job), 200) if job else (jsonify({"error": "Unknown job."}), 404)


@app.route("/intake/<uid>/preview.png")
def serve_intake_preview(uid):
    try:
        return send_from_directory(intake._upload_dir(uid), "preview.png")
    except FileNotFoundError:
        return jsonify({"error": "Unknown upload."}), 404


@app.route("/")
@app.route("/<path:filename>")
def serve(filename="workbench.html"):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
