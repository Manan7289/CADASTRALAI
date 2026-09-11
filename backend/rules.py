"""Real geometric compliance rule engine. Every check here is genuine shapely
buffer / intersection / distance computation in local metres (see
geo_utils.LocalProjection) against real OpenStreetMap reference layers and
the real model-extracted building footprints from model.py -- nothing here
is hand-scripted per parcel the way the earlier concept demo was.
"""
import json
from pathlib import Path

from shapely.geometry import shape
from shapely.ops import unary_union

from geo_utils import LocalProjection, load_geojson

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

RAIL_BUFFER_M = 30
WATER_BUFFER_M = 15
# NOTE: OSM's road network in this AOI is itself incomplete (only 49 tagged ways for
# a town this size -- confirmed by the same kind of manual check that found the
# building-tagging gap). A strict real-world threshold (e.g. 25m) would flag ~94% of
# buildings purely from missing road data, not genuine lack of access. 200m instead
# flags roughly the worst quartile -- read this as "relatively road-poor given what's
# mapped", not a definitive no-access finding.
ROAD_CONNECT_M = 200

CITATIONS = {
    "RAIL_BUFFER": "Railways Act, 1989 - Sec. 3 (30 m safety strip)",
    "WATER_BUFFER": "State River Regulation Zone norms (15 m setback)",
    "GOVT_ENCROACH": "Revenue Dept. review - structure on classified government/institutional land",
    "NO_ROAD": "Relative to the mapped OSM road network (itself incomplete here) - review access in the field before treating as a genuine easement issue",
    "UNRECORDED": "Structure visible in imagery but absent from the OSM/cadastral building layer - verify permit",
}


def _polys_only(fc):
    return [shape(f["geometry"]) for f in fc["features"] if f["geometry"]["type"] == "Polygon"]


def _lines_only(fc):
    return [shape(f["geometry"]) for f in fc["features"] if f["geometry"]["type"] == "LineString"]


def load_layers():
    buildings_fc = load_geojson(PROC_DIR / "extracted_buildings.geojson")
    roads_fc = load_geojson(PROC_DIR / "roads.geojson")
    rail_fc = load_geojson(PROC_DIR / "railway.geojson")
    water_fc = load_geojson(PROC_DIR / "waterway.geojson")
    govt_fc = load_geojson(PROC_DIR / "government.geojson")

    all_coords = []
    for f in buildings_fc["features"]:
        all_coords.extend(f["geometry"]["coordinates"][0])
    if not all_coords:
        raise RuntimeError("No extracted buildings found -- run model.py first")
    lon0 = sum(c[0] for c in all_coords) / len(all_coords)
    lat0 = sum(c[1] for c in all_coords) / len(all_coords)
    proj = LocalProjection(lon0, lat0)

    return {
        "proj": proj,
        "buildings_fc": buildings_fc,
        "buildings": [(f["properties"], proj.to_local(shape(f["geometry"]))) for f in buildings_fc["features"]],
        "roads": [proj.to_local(g) for g in _lines_only(roads_fc)],
        "rail": [proj.to_local(g) for g in _lines_only(rail_fc)],
        "water_lines": [proj.to_local(g) for g in _lines_only(water_fc)],
        "water_polys": [proj.to_local(g) for g in _polys_only(water_fc)],
        "govt": [proj.to_local(g) for g in _polys_only(govt_fc)],
    }


def evaluate_all():
    layers = load_layers()
    rail_lines_union = unary_union(layers["rail"]) if layers["rail"] else None
    rail_union = rail_lines_union.buffer(RAIL_BUFFER_M) if rail_lines_union is not None else None
    water_geoms = layers["water_lines"] + layers["water_polys"]
    water_geoms_union = unary_union(water_geoms) if water_geoms else None
    water_union = water_geoms_union.buffer(WATER_BUFFER_M) if water_geoms_union is not None else None
    govt_union = unary_union(layers["govt"]) if layers["govt"] else None
    road_union = unary_union(layers["roads"]) if layers["roads"] else None

    results = []
    for props, geom in layers["buildings"]:
        alerts = []

        if rail_union is not None and geom.intersects(rail_union):
            dist = geom.distance(rail_lines_union)
            alerts.append({"type": "RAIL_BUFFER", "msg": f"Footprint is ~{dist:.0f} m from the rail line - inside the {RAIL_BUFFER_M} m safety buffer.", "citation": CITATIONS["RAIL_BUFFER"]})

        if water_union is not None and geom.intersects(water_union):
            dist = geom.distance(water_geoms_union)
            alerts.append({"type": "WATER_BUFFER", "msg": f"Footprint is ~{dist:.0f} m from a waterway/water body - inside the {WATER_BUFFER_M} m buffer.", "citation": CITATIONS["WATER_BUFFER"]})

        if govt_union is not None and geom.intersects(govt_union):
            overlap_area = geom.intersection(govt_union).area
            if overlap_area > 1.0:
                alerts.append({"type": "GOVT_ENCROACH", "msg": f"Structure overlaps government/institutional land by ~{overlap_area:.0f} m2.", "citation": CITATIONS["GOVT_ENCROACH"]})

        if road_union is not None:
            d = geom.distance(road_union)
            if d > ROAD_CONNECT_M:
                alerts.append({"type": "NO_ROAD", "msg": f"Nearest mapped road is ~{d:.0f} m away.", "citation": CITATIONS["NO_ROAD"]})

        if props.get("unrecorded"):
            alerts.append({"type": "UNRECORDED", "msg": "Detected in imagery; no matching footprint in the existing building layer.", "citation": CITATIONS["UNRECORDED"]})

        results.append({"id": props.get("id"), "alerts": alerts, "area_m2": round(geom.area, 1)})

    return results


if __name__ == "__main__":
    res = evaluate_all()
    flagged = [r for r in res if r["alerts"]]
    print(f"{len(res)} buildings evaluated, {len(flagged)} flagged")
    for r in flagged[:15]:
        print(r["id"], [a["type"] for a in r["alerts"]])
