"""Reference data for a survey: existing GIS parcel layers and GNSS / ground-
truth points, compared against the AI-extracted parcels.

  data/surveys/<id>/reference/
    parcels.geojson   existing parcel layer, EPSG:4326, original attributes kept
    gnss.geojson      GNSS/CORS or field ground-truth points, EPSG:4326
    meta.json         what was imported, from which files, what was dropped
    comparison.json   latest comparison result (summary + per-feature detail)

Parcel layers: GeoJSON, GeoPackage, zipped Shapefile or KML -- any CRS, as
long as the file declares it. GNSS points: CSV with latitude/longitude, or
easting/northing plus the EPSG code of the survey data (e.g. UTM 44N =
32644), or a GeoJSON of points.
"""
import csv
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Point, box, mapping

import survey

LAT_KEYS = ("lat", "latitude", "y_lat")
LON_KEYS = ("lon", "lng", "long", "longitude", "x_lon")
EAST_KEYS = ("easting", "east", "x", "e")
NORTH_KEYS = ("northing", "north", "y", "n")
ID_KEYS = ("id", "point_id", "name", "pt", "point")
TYPE_KEYS = ("type", "code", "kind", "desc", "description")
ACC_KEYS = ("hrms", "accuracy", "acc", "h_acc", "horizontal_accuracy", "sigma")


def ref_dir(sid):
    d = survey.survey_dir(sid) / "reference"
    d.mkdir(exist_ok=True)
    return d


def _meta(sid):
    p = ref_dir(sid) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_meta(sid, **kw):
    m = _meta(sid)
    m.update(kw)
    (ref_dir(sid) / "meta.json").write_text(json.dumps(m, indent=2))
    return m


def _survey_box(sid):
    meta = json.loads((survey.survey_dir(sid) / "meta.json").read_text())
    (s, w), (n, e) = meta["bounds"]
    return box(w, s, e, n)


def _read_vector(file_storage):
    name = Path(file_storage.filename).name.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        file_storage.save(path)
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)
            shp = sorted(Path(tmp).rglob("*.shp")) + sorted(Path(tmp).rglob("*.gpkg")) + sorted(Path(tmp).rglob("*.geojson"))
            if not shp:
                raise ValueError("The zip file has no .shp, .gpkg or .geojson inside.")
            path = shp[0]
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        if name.endswith((".geojson", ".json")):
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError("The layer has no coordinate reference system (for a Shapefile, include the .prj).")
    return gdf


def import_parcels(sid, file_storage, id_field=None):
    gdf = _read_vector(file_storage).to_crs("EPSG:4326")
    n_in = len(gdf)
    gdf["_row"] = range(len(gdf))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"]
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"]
    inside = gdf[gdf.intersects(_survey_box(sid))]
    if inside.empty:
        raise ValueError("None of the parcels in that layer fall inside this survey's area.")

    feats = []
    for i, (_, row) in enumerate(inside.iterrows(), 1):
        props = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                 for k, v in row.drop(labels=["geometry", "_row"]).items()}
        ref_id = str(props.get(id_field)) if id_field and props.get(id_field) is not None else str(i)
        feats.append({"type": "Feature", "properties": {"ref_id": ref_id, "attrs": props},
                      "geometry": mapping(row.geometry)})
    (ref_dir(sid) / "parcels.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return _save_meta(sid, parcels={"file": Path(file_storage.filename).name, "features_in_file": n_in,
                                    "imported_polygons": len(feats), "records_used": int(inside["_row"].nunique()),
                                    "records_skipped": n_in - int(inside["_row"].nunique()),
                                    "id_field": id_field, "imported_at": time.strftime("%Y-%m-%d %H:%M")})


def _pick(fieldnames, keys):
    low = {f.lower().strip(): f for f in fieldnames}
    for k in keys:
        if k in low:
            return low[k]
    return None


def import_gnss(sid, file_storage, epsg=None):
    name = Path(file_storage.filename).name.lower()
    points = []
    if name.endswith((".geojson", ".json")):
        gdf = gpd.read_file(io.BytesIO(file_storage.read()))
        gdf = (gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf).to_crs("EPSG:4326")
        for i, (_, row) in enumerate(gdf[gdf.geom_type == "Point"].iterrows(), 1):
            props = row.drop(labels="geometry").to_dict()
            points.append((row.geometry.x, row.geometry.y, str(props.get("id", i)), str(props.get("type", "corner")),
                           props.get("accuracy")))
    else:
        text = file_storage.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []
        lat_f, lon_f = _pick(fields, LAT_KEYS), _pick(fields, LON_KEYS)
        east_f, north_f = _pick(fields, EAST_KEYS), _pick(fields, NORTH_KEYS)
        id_f, type_f, acc_f = _pick(fields, ID_KEYS), _pick(fields, TYPE_KEYS), _pick(fields, ACC_KEYS)
        if lat_f and lon_f:
            to_ll = None
        elif east_f and north_f:
            if not epsg:
                raise ValueError("Easting/northing columns found: give the EPSG code of those coordinates (e.g. 32644 for UTM 44N).")
            to_ll = Transformer.from_crs(CRS.from_epsg(int(epsg)), "EPSG:4326", always_xy=True)
        else:
            raise ValueError(f"Could not find latitude/longitude or easting/northing columns in: {', '.join(fields)}")
        for i, r in enumerate(reader, 1):
            try:
                if to_ll is None:
                    lon, lat = float(r[lon_f]), float(r[lat_f])
                else:
                    lon, lat = to_ll.transform(float(r[east_f]), float(r[north_f]))
            except (TypeError, ValueError):
                continue
            acc = None
            if acc_f and r.get(acc_f):
                try:
                    acc = float(r[acc_f])
                except ValueError:
                    acc = None
            points.append((lon, lat, r.get(id_f) or str(i) if id_f else str(i),
                           (r.get(type_f) or "corner") if type_f else "corner", acc))

    area = _survey_box(sid)
    feats = [{"type": "Feature", "properties": {"point_id": pid, "type": str(t).strip().lower() or "corner", "accuracy_m": acc},
              "geometry": {"type": "Point", "coordinates": [lon, lat]}}
             for lon, lat, pid, t, acc in points if area.buffer(0.0005).contains(Point(lon, lat))]
    if not feats:
        raise ValueError("No usable points inside this survey's area.")
    (ref_dir(sid) / "gnss.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return _save_meta(sid, gnss={"file": Path(file_storage.filename).name, "rows": len(points), "imported": len(feats),
                                 "outside_survey": len(points) - len(feats), "epsg": epsg,
                                 "imported_at": time.strftime("%Y-%m-%d %H:%M")})


def clear(sid, kind):
    d = ref_dir(sid)
    for name in ({"parcels": ["parcels.geojson"], "gnss": ["gnss.geojson"]}[kind] + ["comparison.json"]):
        (d / name).unlink(missing_ok=True)
    if kind == "parcels":
        # the comparison against that record no longer applies
        parcels_path = survey.survey_dir(sid) / "parcels.geojson"
        fc = json.loads(parcels_path.read_text())
        for f in fc["features"]:
            for k in ("record_status", "record_iou", "record_ref_id"):
                f["properties"].pop(k, None)
        parcels_path.write_text(json.dumps(fc))
    m = _meta(sid)
    m.pop(kind, None)
    (d / "meta.json").write_text(json.dumps(m, indent=2))
    return m


def status(sid):
    d = ref_dir(sid)
    cmp_path = d / "comparison.json"
    return {"meta": _meta(sid), "has_parcels": (d / "parcels.geojson").exists(), "has_gnss": (d / "gnss.geojson").exists(),
            "comparison": json.loads(cmp_path.read_text()) if cmp_path.exists() else None}
