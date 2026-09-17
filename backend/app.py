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
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import upload_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
RAW_UPLOADS_DIR = BASE_DIR / "data" / "raw" / "uploads"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB, generous for a short drone video


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


# Training rewrites these every epoch under data/models/, but the frontend reads
# frontend/data/, which only changes when build_static.py runs. Locally, prefer
# the live copy so the Model Output page tracks a run in progress instead of
# looking frozen for an hour. Vercel has no Flask, so there it is just the baked
# file, which is correct -- nothing is training in production.
LIVE_METRICS = {"inria_metrics.json", "uavid_metrics.json"}
MODELS_DIR = BASE_DIR / "data" / "models"


@app.route("/data/<name>")
def serve_data(name):
    if name in LIVE_METRICS and (MODELS_DIR / name).exists():
        return send_from_directory(MODELS_DIR, name)
    return send_from_directory(FRONTEND_DIR / "data", name)


@app.route("/")
@app.route("/<path:filename>")
def serve(filename="index.html"):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
