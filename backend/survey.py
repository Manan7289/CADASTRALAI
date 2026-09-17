"""A survey = one processed drone/aerial acquisition, stored as a folder the
Web-GIS workbench reads and edits:

  data/surveys/<id>/
    meta.json            name, source, CRS, bounds, GSD, model + height usage, stats
    ori.webp             orthoimage warped to Web Mercator for display (+ alpha)
    classes.png          segmentation overlay on the same grid
    height.png           nDSM overlay (only when the survey had a DSM)
    parcels.geojson      EPSG:4326, editable master layer (status, confidence, issues)
    buildings.geojson    roof footprints
    corridors.geojson    roads / lanes / access corridors
    issues.geojson       current topology issues (regenerated on every save)
    edits.jsonl          audit log of every save / auto-fix / merge

Validation and repair always run in the survey's UTM zone (metres).
"""
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform, unary_union

import topology

SURVEYS_DIR = Path(__file__).resolve().parent.parent / "data" / "surveys"
DISPLAY_MAX_PX = 2400
CLASS_COLOURS = {0: (200, 90, 90), 1: (70, 110, 230), 2: (225, 225, 225), 3: (130, 215, 200), 4: (40, 160, 70)}
HEIGHT_DISPLAY_MIN_M = 1.0  # below this the nDSM is mostly ground noise; not drawn
STATUSES = ("draft", "approved", "rejected", "field_check")


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def survey_dir(sid):
    d = SURVEYS_DIR / sid
    if not d.is_dir() or "/" in sid or ".." in sid:
        raise FileNotFoundError(sid)
    return d


def list_surveys():
    out = []
    for d in sorted(SURVEYS_DIR.glob("*/meta.json")):
        m = _read_json(d)
        out.append({k: m.get(k) for k in ("id", "name", "source", "created", "used_height", "stats", "review")})
    return out


def _to_display(arr_hwc, src_crs, src_transform, resampling):
    """Warp an H×W×C uint8 raster to EPSG:3857 (axis-aligned with lat/lon, so
    a Leaflet imageOverlay lines up exactly) at no more than DISPLAY_MAX_PX."""
    h, w, c = arr_hwc.shape
    left, bottom, right, top = array_bounds(h, w, src_transform)
    dst_t, dw, dh = calculate_default_transform(src_crs, "EPSG:3857", w, h, left, bottom, right, top)
    scale = max(dw, dh) / DISPLAY_MAX_PX
    if scale > 1:
        dst_t = dst_t * dst_t.scale(scale)
        dw, dh = int(np.ceil(dw / scale)), int(np.ceil(dh / scale))
    out = np.zeros((c, dh, dw), np.uint8)
    for k in range(c):
        reproject(np.ascontiguousarray(arr_hwc[..., k]), out[k], src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_t, dst_crs="EPSG:3857", resampling=resampling, src_nodata=0, dst_nodata=0)
    l, b, r, t = array_bounds(dh, dw, dst_t)
    to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon0, lat0 = to_ll.transform(l, b)
    lon1, lat1 = to_ll.transform(r, t)
    return out.transpose(1, 2, 0), [[lat0, lon0], [lat1, lon1]]


WEBMERC_HALF = 20037508.342789244
TILE_PX = 256


def _tile_bounds_3857(z, x, y):
    size = 2 * WEBMERC_HALF / 2 ** z
    left, top = -WEBMERC_HALF + x * size, WEBMERC_HALF - y * size
    return left, top - size, left + size, top


def write_tiles(rgba_hwc, src_crs, src_transform, out_dir, gsd_m):
    """XYZ tile pyramid (Web Mercator, 256 px WebP) of the survey orthoimage,
    so the workbench shows full survey resolution at every zoom instead of
    one downscaled overlay. Max zoom is the first level whose pixel is at
    least as fine as the survey GSD; empty tiles are skipped."""
    h, w, _ = rgba_hwc.shape
    left, bottom, right, top = array_bounds(h, w, src_transform)
    to_merc = Transformer.from_crs(src_crs, "EPSG:3857", always_xy=True)
    xs, ys = to_merc.transform([left, right, left, right], [bottom, bottom, top, top])
    mx0, mx1, my0, my1 = min(xs), max(xs), min(ys), max(ys)
    lat = np.degrees(np.arctan(np.sinh(((my0 + my1) / 2) / 6378137.0)))
    res0 = 2 * WEBMERC_HALF / TILE_PX * np.cos(np.radians(lat))  # ground metres per pixel at zoom 0
    zmax = int(min(23, np.ceil(np.log2(res0 / gsd_m))))
    zmin = max(0, zmax - 6)
    bands = [np.ascontiguousarray(rgba_hwc[..., k]) for k in range(4)]
    count = 0
    for z in range(zmin, zmax + 1):
        size = 2 * WEBMERC_HALF / 2 ** z
        tx0, tx1 = int((mx0 + WEBMERC_HALF) // size), int((mx1 + WEBMERC_HALF) // size)
        ty0, ty1 = int((WEBMERC_HALF - my1) // size), int((WEBMERC_HALF - my0) // size)
        resampling = Resampling.bilinear if z == zmax else Resampling.average
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                l, b, r, t = _tile_bounds_3857(z, tx, ty)
                dst_t = from_bounds(l, b, r, t, TILE_PX, TILE_PX)
                tile = np.zeros((4, TILE_PX, TILE_PX), np.uint8)
                for k in range(4):
                    reproject(bands[k], tile[k], src_transform=src_transform, src_crs=src_crs, dst_transform=dst_t,
                              dst_crs="EPSG:3857", resampling=resampling, src_nodata=None, dst_nodata=0)
                if not tile[3].any():
                    continue
                p = out_dir / str(z) / str(tx)
                p.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile.transpose(1, 2, 0), "RGBA").save(p / f"{ty}.webp", quality=80, method=4)
                count += 1
    return {"min_zoom": zmin, "max_zoom": zmax, "count": count}


def _geom_to_ll(geom, transformer):
    return shp_transform(transformer.transform, geom)


def create(name, source, loaded, probs, extracted, model_info):
    """loaded: segment.load_survey() dict; probs: model output; extracted:
    parcel_extract.extract() output. Writes the survey folder, returns meta."""
    sid = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6]
    d = SURVEYS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    crs, transform, valid = loaded["crs"], loaded["transform"], loaded["valid"]
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    alpha = (valid * 255).astype(np.uint8)
    ori, bounds = _to_display(np.dstack([loaded["rgb"], alpha]), crs, transform, Resampling.average)
    Image.fromarray(ori, "RGBA").save(d / "ori.webp", quality=82, method=4)
    tiles = write_tiles(np.dstack([loaded["rgb"], alpha]), crs, transform, d / "tiles", extracted["stats"]["gsd_m"])

    labels = extracted["rasters"]["labels"]
    cls_rgb = np.zeros(labels.shape + (4,), np.uint8)
    for k, col in CLASS_COLOURS.items():
        cls_rgb[labels == k] = col + (255,)
    cls_rgb[~valid, 3] = 0
    cls_img, _ = _to_display(cls_rgb, crs, transform, Resampling.nearest)
    Image.fromarray(cls_img, "RGBA").save(d / "classes.png", optimize=True)

    if loaded.get("ndsm") is not None:
        h = np.clip(loaded["ndsm"] / 15.0, 0, 1)
        ramp = np.dstack([255 * h, 180 * (1 - np.abs(h - 0.5) * 2), 255 * (1 - h), np.where(valid & (loaded["ndsm"] > HEIGHT_DISPLAY_MIN_M), 220, 0)])
        h_img, _ = _to_display(ramp.astype(np.uint8), crs, transform, Resampling.bilinear)
        Image.fromarray(h_img, "RGBA").save(d / "height.png", optimize=True)

    def fc(features):
        return {"type": "FeatureCollection", "features": features}

    parcels = []
    for i, p in enumerate(sorted(extracted["parcels"], key=lambda p: (-p["geometry"].centroid.y, p["geometry"].centroid.x)), 1):
        props = {k: v for k, v in p.items() if k != "geometry"}
        props.update({"id": i, "prov_pin": f"{sid.upper()}-{i:04d}", "status": "draft", "issues": [], "source": "ai"})
        parcels.append({"type": "Feature", "properties": props, "geometry": mapping(_geom_to_ll(p["geometry"], to_ll))})
    buildings = [{"type": "Feature", "properties": {k: v for k, v in b.items() if k != "geometry"},
                  "geometry": mapping(_geom_to_ll(b["geometry"], to_ll))} for b in extracted["buildings"]]
    corridors = [{"type": "Feature", "properties": {k: v for k, v in c.items() if k != "geometry"},
                  "geometry": mapping(_geom_to_ll(c["geometry"], to_ll))} for c in extracted["corridors"]]
    _write_json(d / "buildings.geojson", fc(buildings))
    _write_json(d / "corridors.geojson", fc(corridors))

    meta = {
        "id": sid, "name": name, "source": source, "created": time.strftime("%Y-%m-%d %H:%M"),
        "crs": crs.to_string(), "gsd_m": extracted["stats"]["gsd_m"], "bounds": bounds,
        "used_height": loaded.get("ndsm") is not None, "has_height_layer": (d / "height.png").exists(),
        "tiles": tiles,
        "model": model_info, "stats": extracted["stats"],
    }
    _write_json(d / "meta.json", meta)
    save_parcels(sid, fc(parcels), action="created by AI pipeline")
    return _read_json(d / "meta.json")


def _utm(sid):
    meta = _read_json(survey_dir(sid) / "meta.json")
    fwd = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    back = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True)
    return meta, fwd, back


def _corridor_union_utm(sid, fwd):
    fc = _read_json(survey_dir(sid) / "corridors.geojson")
    geoms = [shp_transform(fwd.transform, shape(f["geometry"])) for f in fc["features"]]
    return unary_union(geoms) if geoms else None


HISTORY_KEEP = 50
TRASH_DIR = SURVEYS_DIR.parent / "surveys_trash"


def _snapshot(d, action):
    """Keep the parcel layer as it was before this save, so it can be undone."""
    src = d / "parcels.geojson"
    if not src.exists():
        return
    hist = d / "history"
    hist.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:4]}"
    (hist / f"{stamp}.geojson").write_bytes(src.read_bytes())
    (hist / f"{stamp}.json").write_text(json.dumps({"before": action}))
    snaps = sorted(hist.glob("*.geojson"))
    for old in snaps[:-HISTORY_KEEP]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


def save_parcels(sid, parcels_fc, action="edited", snapshot=True):
    """Validate, annotate each parcel with its issues, write layer + issues,
    update review stats, append to the audit log. Returns (parcels_fc, issues_fc)."""
    d = survey_dir(sid)
    if snapshot:
        _snapshot(d, action)
    meta, fwd, back = _utm(sid)
    feats = parcels_fc["features"]
    for f in feats:
        f["properties"].setdefault("status", "draft")
        f["properties"].setdefault("source", "manual")
        if not f["properties"].get("prov_pin"):
            f["properties"]["prov_pin"] = f"{sid.upper()}-{int(f['properties']['id']):04d}"
        if f["properties"]["status"] not in STATUSES:
            f["properties"]["status"] = "draft"
    utm = [{"id": f["properties"]["id"], "geometry": shp_transform(fwd.transform, shape(f["geometry"]))} for f in feats]
    for f, u in zip(feats, utm):
        f["properties"]["area_m2"] = round(u["geometry"].area, 1)
        f["properties"]["perimeter_m"] = round(u["geometry"].length, 1)

    issues = topology.validate(utm, exclude=_corridor_union_utm(sid, fwd))
    by_parcel = {}
    issue_feats = []
    for n, it in enumerate(issues, 1):
        for pid in it["parcel_ids"]:
            by_parcel.setdefault(pid, []).append(it["type"])
        issue_feats.append({"type": "Feature", "geometry": mapping(shp_transform(back.transform, it["geometry"])),
                            "properties": {"n": n, "type": it["type"], "severity": it["severity"],
                                           "parcel_ids": it["parcel_ids"], "message": it["message"],
                                           "suggested_fix": it["suggested_fix"]}})
    for f in feats:
        f["properties"]["issues"] = sorted(set(by_parcel.get(f["properties"]["id"], [])))

    issues_fc = {"type": "FeatureCollection", "features": issue_feats}
    _write_json(d / "parcels.geojson", parcels_fc)
    _write_json(d / "issues.geojson", issues_fc)

    counts = {s: 0 for s in STATUSES}
    for f in feats:
        counts[f["properties"]["status"]] += 1
    meta["review"] = {"parcels": len(feats), **counts, "issues": topology.summarize(issues)}
    _write_json(d / "meta.json", meta)
    with open(d / "edits.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
                              "parcels": len(feats), "issues": len(issues)}) + "\n")
    return parcels_fc, issues_fc


def auto_fix(sid):
    d = survey_dir(sid)
    meta, fwd, back = _utm(sid)
    fc = _read_json(d / "parcels.geojson")
    props = {f["properties"]["id"]: f["properties"] for f in fc["features"]}
    def priority(p):
        # a surveyor's own work outranks the AI when auto-fix has to pick a winner
        if p.get("status") == "approved" or p.get("source") in ("manual", "edited"):
            return 2.0
        return p.get("confidence") or 0.0

    utm = [{"id": f["properties"]["id"], "geometry": shp_transform(fwd.transform, shape(f["geometry"])),
            "confidence": priority(f["properties"])} for f in fc["features"]]
    fixed, log = topology.auto_fix(utm, exclude=_corridor_union_utm(sid, fwd))
    new_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props[p["id"]], "geometry": mapping(shp_transform(back.transform, p["geometry"]))}
        for p in fixed if not p["geometry"].is_empty]}
    parcels_fc, issues_fc = save_parcels(sid, new_fc, action=f"auto-fix: {len(log)} changes")
    return parcels_fc, issues_fc, log


def merge(sid, ids):
    d = survey_dir(sid)
    meta, fwd, back = _utm(sid)
    fc = _read_json(d / "parcels.geojson")
    ids = [int(i) for i in ids]
    chosen = [f for f in fc["features"] if f["properties"]["id"] in ids]
    if len(chosen) < 2:
        raise ValueError("Select at least two parcels to merge.")
    merged = unary_union([shp_transform(fwd.transform, shape(f["geometry"])) for f in chosen])
    if merged.geom_type != "Polygon":
        raise ValueError("Those parcels don't share a boundary, so merging them would create a multipart parcel.")
    keep = max(chosen, key=lambda f: shape(f["geometry"]).area)
    keep["geometry"] = mapping(shp_transform(back.transform, merged))
    keep["properties"].update({"status": "draft", "source": "edited"})
    fc["features"] = [f for f in fc["features"] if f["properties"]["id"] not in ids or f is keep]
    return save_parcels(sid, fc, action=f"merged parcels {ids} into {keep['properties']['id']}")


def undo(sid):
    """Restore the parcel layer from before the most recent change."""
    d = survey_dir(sid)
    snaps = sorted((d / "history").glob("*.geojson")) if (d / "history").exists() else []
    if not snaps:
        raise ValueError("Nothing to undo.")
    last = snaps[-1]
    undone = json.loads(last.with_suffix(".json").read_text()).get("before", "last change") if last.with_suffix(".json").exists() else "last change"
    fc = _read_json(last)
    last.unlink()
    last.with_suffix(".json").unlink(missing_ok=True)
    parcels_fc, issues_fc = save_parcels(sid, fc, action=f"undo: {undone}", snapshot=False)
    return parcels_fc, issues_fc, undone


def history(sid, limit=200):
    d = survey_dir(sid)
    log = d / "edits.jsonl"
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.exists() else []
    undoable = len(list((d / "history").glob("*.geojson"))) if (d / "history").exists() else 0
    return {"entries": entries[-limit:][::-1], "undo_available": undoable}


def rename(sid, name):
    name = (name or "").strip()[:120]
    if not name:
        raise ValueError("Name cannot be empty.")
    d = survey_dir(sid)
    meta = _read_json(d / "meta.json")
    old = meta.get("name")
    meta["name"] = name
    _write_json(d / "meta.json", meta)
    with open(d / "edits.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "action": f"renamed from '{old}' to '{name}'"}) + "\n")
    return meta


def delete(sid):
    """Soft delete: the survey folder moves to data/surveys_trash/ and can be restored by moving it back."""
    import shutil
    d = survey_dir(sid)
    TRASH_DIR.mkdir(exist_ok=True)
    dest = TRASH_DIR / f"{sid}--deleted-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(str(d), str(dest))
    return {"id": sid, "moved_to": str(dest.relative_to(SURVEYS_DIR.parent.parent))}
