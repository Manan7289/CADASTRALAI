"""CadastraAI web server: the Survey Workbench, New Survey intake, Records
comparison and Field verification pages, and their JSON API.

Run:   python backend/app.py                 -> http://127.0.0.1:5050
Phone: CADASTRAAI_HTTPS=1 python backend/app.py  -> https://<LAN-IP>:5443 (GPS needs HTTPS)
"""
import json
import traceback
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

import compare
import field
import intake
import reference
import segment
import survey
from export import export as export_parcels

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # orthomosaics and DSMs are large


# ---------------------------------------------------------------- survey workbench API
@app.route("/api/surveys")
def api_surveys():
    return jsonify(survey.list_surveys())


@app.route("/api/surveys/<sid>", methods=["PATCH", "DELETE"])
def api_survey_manage(sid):
    try:
        if request.method == "DELETE":
            return jsonify(survey.delete(sid))
        return jsonify(survey.rename(sid, (request.get_json(silent=True) or {}).get("name")))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/surveys/<sid>/history")
def api_history(sid):
    try:
        return jsonify(survey.history(sid))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404


@app.route("/api/surveys/<sid>/undo", methods=["POST"])
def api_undo(sid):
    try:
        parcels_fc, issues_fc, undone = survey.undo(sid)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"parcels": parcels_fc, "issues": issues_fc, "undone": undone})


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
        d = survey.survey_dir(sid)
        read = lambda f: json.loads((d / f).read_text(encoding="utf-8")) if (d / f).exists() else None
        data, name, mime = export_parcels(read("parcels.geojson"), fmt, read("buildings.geojson"), read("corridors.geojson"),
                                           read("roads.geojson"))
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


# ---------------------------------------------------------------- reference data + comparison
@app.route("/api/surveys/<sid>/reference")
def api_reference_status(sid):
    try:
        return jsonify(reference.status(sid))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404


@app.route("/api/surveys/<sid>/reference/<kind>", methods=["POST", "DELETE"])
def api_reference_import(sid, kind):
    if kind not in ("parcels", "gnss"):
        return jsonify({"error": "kind must be parcels or gnss."}), 400
    try:
        if request.method == "DELETE":
            reference.clear(sid, kind)
            return jsonify(reference.status(sid))
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "No file uploaded."}), 400
        if kind == "parcels":
            reference.import_parcels(sid, f, id_field=(request.form.get("id_field") or "").strip() or None)
        else:
            reference.import_gnss(sid, f, epsg=(request.form.get("epsg") or "").strip() or None)
        return jsonify(reference.status(sid))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not read that file: {e}"}), 400


@app.route("/api/surveys/<sid>/compare", methods=["POST"])
def api_compare(sid):
    body = request.get_json(silent=True) or {}
    try:
        tol = float(body.get("tolerance_m", 1.0))
        result = compare.compare(sid, tolerance_m=tol)
        compare.annotate_parcels(sid, result)
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


# ---------------------------------------------------------------- field verification
@app.route("/api/surveys/<sid>/field")
def api_field(sid):
    try:
        pid = request.args.get("parcel_id")
        return jsonify({"summary": field.summary(sid),
                        "observations": field.observations(sid, int(pid) if pid else None)})
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404


@app.route("/api/surveys/<sid>/field/<int:parcel_id>", methods=["POST"])
def api_field_record(sid, parcel_id):
    try:
        obs, parcels_fc, issues_fc = field.record(sid, parcel_id, request.form, request.files.getlist("photos"))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"observation": obs, "parcels": parcels_fc, "issues": issues_fc})


@app.route("/api/surveys/<sid>/field/corner", methods=["POST"])
def api_field_corner(sid):
    try:
        return jsonify(field.record_corner(sid, request.form))
    except FileNotFoundError:
        return jsonify({"error": "Unknown survey."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


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
        job_id = intake.start_job(uid, (body.get("name") or "").strip()[:120], aoi, body.get("model") or "auto")
    except FileNotFoundError:
        return jsonify({"error": "Unknown upload."}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"job_id": job_id})


@app.route("/api/models")
def api_models():
    return jsonify({k: {kk: v[kk] for kk in ("label", "summary", "limits")} for k, v in segment.MODELS.items()})


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
    import os
    # CADASTRAAI_HTTPS=1 serves on the local network over HTTPS (self-signed), which phone
    # browsers require before they will share GPS with the field verification page
    if os.environ.get("CADASTRAAI_HTTPS") == "1":
        app.run(host="0.0.0.0", port=5443, ssl_context="adhoc", debug=False)
    else:
        app.run(debug=True, port=5050)
