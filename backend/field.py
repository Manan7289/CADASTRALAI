"""Field verification: what a surveyor records on site for a flagged parcel.

  data/surveys/<id>/field/
    observations.jsonl   one JSON object per visit (append-only)
    photos/<obs>_<n>.jpg re-encoded, metadata stripped, max 1600 px

A visit records a verdict, a note, optional photos and the phone's GPS fix
(with its reported accuracy). The server computes how far that fix was from
the parcel, so a reviewer can tell an on-site check from a desk check.

Verdict -> parcel review status (applied through survey.save_parcels, so
topology is re-validated and the edit log records it):
  confirmed           -> approved
  boundary_wrong      -> field_check  (stays open: the boundary needs editing)
  not_a_parcel        -> rejected
  part_of_neighbour   -> field_check  (stays open: needs a merge)
  could_not_access    -> field_check

A surveyor can also record a boundary corner at their current position; it
is appended to the survey's GNSS reference layer (type 'corner', with the
phone's accuracy) and so feeds the Records comparison.
"""
import io
import json
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform

import reference
import survey

VERDICTS = {
    "confirmed": ("approved", "Boundary confirmed on site"),
    "boundary_wrong": ("field_check", "Boundary is wrong -- needs editing"),
    "not_a_parcel": ("rejected", "Not a real parcel"),
    "part_of_neighbour": ("field_check", "Part of a neighbouring parcel -- needs merging"),
    "could_not_access": ("field_check", "Could not access the site"),
}
MAX_PHOTOS = 4
MAX_PHOTO_BYTES = 12 * 1024 * 1024
PHOTO_MAX_PX = 1600
ON_SITE_M = 30


def field_dir(sid):
    d = survey.survey_dir(sid) / "field"
    (d / "photos").mkdir(parents=True, exist_ok=True)
    return d


def observations(sid, parcel_id=None):
    path = field_dir(sid) / "observations.jsonl"
    if not path.exists():
        return []
    out = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [o for o in out if parcel_id is None or o["parcel_id"] == parcel_id]


def _save_photo(d, obs_id, n, file_storage):
    data = file_storage.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError("Photo is larger than 12 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")   # respect phone orientation, then drop EXIF (incl. GPS)
    except Exception:
        raise ValueError(f"{file_storage.filename} is not a readable image.")
    img.thumbnail((PHOTO_MAX_PX, PHOTO_MAX_PX))
    name = f"{obs_id}_{n}.jpg"
    img.save(d / "photos" / name, "JPEG", quality=82)
    return name


def _gps(form):
    try:
        lat, lon = float(form["lat"]), float(form["lon"])
    except (KeyError, ValueError):
        return None
    acc = form.get("accuracy")
    try:
        acc = round(float(acc), 1) if acc not in (None, "") else None
    except ValueError:
        acc = None
    return {"lat": lat, "lon": lon, "accuracy_m": acc}


def record(sid, parcel_id, form, photos):
    verdict = form.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("Choose a verdict.")
    fc = json.loads((survey.survey_dir(sid) / "parcels.geojson").read_text(encoding="utf-8"))
    feat = next((f for f in fc["features"] if f["properties"]["id"] == parcel_id), None)
    if feat is None:
        raise ValueError("Unknown parcel.")
    photos = [p for p in photos if p and p.filename][:MAX_PHOTOS]

    meta, fwd, _ = survey._utm(sid)
    gps = _gps(form)
    distance = None
    if gps:
        g = shp_transform(fwd.transform, shape(feat["geometry"]))
        x, y = fwd.transform(gps["lon"], gps["lat"])
        distance = round(g.distance(Point(x, y)), 1)

    d = field_dir(sid)
    obs_id = time.strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:4]
    names = [_save_photo(d, obs_id, i + 1, p) for i, p in enumerate(photos)]
    new_status, label = VERDICTS[verdict]
    obs = {"obs_id": obs_id, "parcel_id": parcel_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "verdict": verdict, "verdict_label": label, "note": (form.get("note") or "").strip()[:2000],
           "surveyor": (form.get("surveyor") or "").strip()[:80], "gps": gps,
           "distance_to_parcel_m": distance, "on_site": distance is not None and distance <= ON_SITE_M,
           "photos": names, "status_set": new_status}
    with open(d / "observations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(obs) + "\n")

    p = feat["properties"]
    p["status"] = new_status
    p["field_verdict"] = verdict
    p["field_visits"] = int(p.get("field_visits") or 0) + 1
    p["field_on_site"] = obs["on_site"]
    parcels_fc, issues_fc = survey.save_parcels(sid, fc, action=f"field verification: parcel {parcel_id} -> {verdict}")
    return obs, parcels_fc, issues_fc


def record_corner(sid, form):
    gps = _gps(form)
    if not gps:
        raise ValueError("No GPS position was sent.")
    rd = reference.ref_dir(sid)
    path = rd / "gnss.geojson"
    fc = json.loads(path.read_text()) if path.exists() else {"type": "FeatureCollection", "features": []}
    n = sum(1 for f in fc["features"] if str(f["properties"].get("point_id", "")).startswith("FIELD-")) + 1
    pid = f"FIELD-{n:03d}"
    fc["features"].append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [gps["lon"], gps["lat"]]},
                           "properties": {"point_id": pid, "type": "corner", "accuracy_m": gps["accuracy_m"],
                                          "source": "phone GPS (field verification)",
                                          "parcel_id": form.get("parcel_id"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}})
    path.write_text(json.dumps(fc))
    meta = reference._meta(sid)
    g = meta.get("gnss") or {"file": "field-recorded corners", "rows": 0, "imported": 0, "outside_survey": 0, "epsg": None}
    g["imported"] = len(fc["features"])
    g["field_recorded"] = n
    reference._save_meta(sid, gnss=g)
    return {"point_id": pid, **gps}


def summary(sid):
    obs = observations(sid)
    by_verdict = {}
    for o in obs:
        by_verdict[o["verdict"]] = by_verdict.get(o["verdict"], 0) + 1
    return {"visits": len(obs), "parcels_visited": len({o["parcel_id"] for o in obs}),
            "on_site": sum(1 for o in obs if o["on_site"]), "by_verdict": by_verdict,
            "verdicts": {k: v[1] for k, v in VERDICTS.items()}}
