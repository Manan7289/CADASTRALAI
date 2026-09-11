"""Fetch real OpenStreetMap vector data for the demo AOI (Igatpuri, Maharashtra
-- a real railway town) and write it to data/processed/*.geojson.

This is real, live, licensed (ODbL) OpenStreetMap data for a real place --
not synthetic. It is used two ways in this project:
  1. As the ward's road / rail / waterway / government-land reference layers
     (the things a real deployment would already have from survey/GIS records).
  2. As ground truth to train and evaluate the building-footprint extraction
     model in model.py.
"""
import json
import time
from pathlib import Path

import requests

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
HEADERS = {"User-Agent": "CadastraAI-SIH2026-Demo/1.0 (educational hackathon project)"}

# Igatpuri, Maharashtra -- real railway town, ~1.5km x 1.6km AOI chosen
# because it genuinely has buildings + roads + railway + waterway + government land.
BBOX = (19.690, 73.555, 19.705, 73.570)  # south, west, north, east

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def overpass(query: str) -> dict:
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                r = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=60)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(3)
    raise RuntimeError(f"Overpass query failed on all endpoints: {last_err}")


def ways_query(bbox, filter_clause):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:60];
    way{filter_clause}({s},{w},{n},{e});
    out geom;
    """


def nodes_query(bbox, filter_clause):
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:60];
    node{filter_clause}({s},{w},{n},{e});
    out;
    """


def way_to_linestring(el):
    coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
    return {"type": "LineString", "coordinates": coords}


def way_to_polygon(el):
    coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Polygon", "coordinates": [coords]}


def is_closed(el):
    g = el.get("geometry")
    return bool(g) and g[0]["lat"] == g[-1]["lat"] and g[0]["lon"] == g[-1]["lon"]


def feature_collection(features):
    return {"type": "FeatureCollection", "features": features}


def fetch_buildings():
    data = overpass(ways_query(BBOX, '["building"]'))
    feats = []
    for el in data["elements"]:
        if el["type"] != "way" or not is_closed(el):
            continue
        feats.append({
            "type": "Feature",
            "properties": {"id": el["id"], "kind": el.get("tags", {}).get("building", "yes")},
            "geometry": way_to_polygon(el),
        })
    return feature_collection(feats)


def fetch_roads():
    data = overpass(ways_query(BBOX, '["highway"]'))
    feats = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        feats.append({
            "type": "Feature",
            "properties": {"id": el["id"], "highway": el.get("tags", {}).get("highway")},
            "geometry": way_to_linestring(el),
        })
    return feature_collection(feats)


def fetch_railway():
    data = overpass(ways_query(BBOX, '["railway"="rail"]'))
    feats = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        feats.append({
            "type": "Feature",
            "properties": {"id": el["id"]},
            "geometry": way_to_linestring(el),
        })
    return feature_collection(feats)


def fetch_waterway():
    s, w, n, e = BBOX
    q = f"""
    [out:json][timeout:60];
    (
      way["waterway"]({s},{w},{n},{e});
      way["natural"="water"]({s},{w},{n},{e});
    );
    out geom;
    """
    data = overpass(q)
    feats = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        geom = way_to_polygon(el) if (tags.get("natural") == "water" and is_closed(el)) else way_to_linestring(el)
        feats.append({"type": "Feature", "properties": {"id": el["id"], **tags}, "geometry": geom})
    return feature_collection(feats)


def fetch_government():
    s, w, n, e = BBOX
    q = f"""
    [out:json][timeout:60];
    (
      way["amenity"~"school|townhall|government_office|community_centre"]({s},{w},{n},{e});
      node["amenity"~"school|townhall|government_office|community_centre"]({s},{w},{n},{e});
    );
    out geom;
    """
    data = overpass(q)
    feats = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way" and is_closed(el):
            geom = way_to_polygon(el)
        elif el["type"] == "node":
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        else:
            continue
        feats.append({"type": "Feature", "properties": {"id": el["id"], "amenity": tags.get("amenity"), "name": tags.get("name")}, "geometry": geom})
    return feature_collection(feats)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    fetchers = {
        "buildings": fetch_buildings,
        "roads": fetch_roads,
        "railway": fetch_railway,
        "waterway": fetch_waterway,
        "government": fetch_government,
    }

    meta = {"bbox": {"south": BBOX[0], "west": BBOX[1], "north": BBOX[2], "east": BBOX[3]}, "source": "OpenStreetMap (ODbL), fetched live via Overpass API", "area": "Igatpuri, Maharashtra, India"}

    for name, fn in fetchers.items():
        print(f"Fetching {name} ...")
        fc = fn()
        out_path = PROC_DIR / f"{name}.geojson"
        out_path.write_text(json.dumps(fc), encoding="utf-8")
        print(f"  -> {len(fc['features'])} features written to {out_path}")
        time.sleep(1.5)  # be polite to the shared Overpass endpoint

    (PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
