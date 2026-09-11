"""Real land-parcel delineation -- the actual "cadastral mapping" step, as
distinct from building-footprint extraction.

Buildings sit inside parcels; a parcel is a plot of land, and OpenStreetMap
(crowd-mapped physical features) simply does not carry land-ownership parcel
boundaries -- that data lives in government cadastral records (Bhu Naksha /
SVAMITVA in India), which aren't a public API. So real parcel boundaries for
an arbitrary AOI can't be "fetched"; they have to be inferred from what IS
observable, which is exactly the approach the UAV-cadastral-mapping
literature this project's own submission cites uses (Crommelinck et al.):

  1. Roads, the railway and water are real, physical dividers of land --
     subtracting their real-world footprint from the AOI leaves the buildable
     land area.
  2. Within that land area, a Voronoi tessellation around each real building
     centroid gives each building the region of land closer to it than to
     any other building -- a standard, published proxy for an individual
     plot boundary when true ownership lines aren't available.
  3. That raw Voronoi cell is then capped to a building's own footprint plus
     a modest yard/setback margin (PLOT_SETBACK_M below). Without this cap,
     an isolated building's cell balloons to fill an entire empty block --
     a single house was coming out with a "residential" parcel over 20
     hectares, which no real cadastral map would ever show. Real plots in a
     built-up area run from under a hundred to a few thousand square metres;
     capping by footprint+setback keeps inferred plots in that range
     regardless of how sparse the neighbouring buildings are.
  4. Whatever land is left over after capping every building's plot in a
     block becomes its own separate vacant/unbuilt parcel (there can be more
     than one per block) -- instead of being silently absorbed into a
     neighbouring building's "residential" plot. If that leftover patch is
     itself large, it's subdivided into a grid of plot/field-sized parcels
     (bigger cells over land classified Agricultural/Forest, smaller cells
     otherwise) rather than left as one big undifferentiated blob -- real
     cadastral maps subdivide farmland into individual survey plots too, not
     just built-up land.

Every parcel then gets real attributes: area and perimeter (dimensions),
land-use (from real OSM landuse tags, or "Residential" if it contains a
building, else "Vacant / Unclassified"), road-frontage connectivity, and the
same rule-engine checks used for buildings, aggregated up to parcel level.
This is an approximation -- it will not match a legal survey -- and the
frontend and report text say so; it is presented as a candidate parcel
layer for verification, matching this project's "preliminary map, human
approves" design.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import shape, Polygon, box, mapping
from shapely.ops import unary_union

from geo_utils import LocalProjection, load_geojson

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ROAD_BUFFER_M = 4       # half-width allowed for an informal road corridor
RAIL_BUFFER_M = 8       # track + ballast width (separate from the 30m safety buffer check)
WATER_BUFFER_M = 3
MIN_PARCEL_M2 = 25      # discard degenerate sliver parcels below this
ROAD_CONNECT_M = 200    # same relaxed threshold as rules.py, for the same reason (sparse OSM roads here)
FAR_MULTIPLIER = 50     # how far outside the AOI the Voronoi "mirror" points sit
PLOT_SETBACK_M = 7      # yard/setback margin added around a building's own footprint to form its plot;
                         # caps how large an inferred residential parcel can get regardless of Voronoi cell size
FIELD_UNSPLIT_M2 = 4000        # tagged farmland/forest under this size stays a single field parcel
FIELD_CELL_MIN_M = 35          # smallest field-subdivision cell
FIELD_CELL_MAX_M = 90          # largest field-subdivision cell
FIELD_TARGET_CELLS = 10        # aim for roughly this many field parcels per tagged patch
UNTAGGED_VACANT_CAP_M2 = 20000 # land with NO landuse tag at all is left as one parcel below this size --
                                # there's no real signal it's several separate plots, so it isn't invented
UNTAGGED_CELL_M = 90            # only used above the cap, to avoid one absurd single blob


def subdivide_grid(poly, cell_size):
    """Cut a polygon into a regular grid of cell_size x cell_size squares,
    clipped to the polygon's real shape -- used so a large stretch of open
    land reads as individual field/plot-sized parcels instead of one blob."""
    minx, miny, maxx, maxy = poly.bounds
    pieces = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            piece = box(x, y, x + cell_size, y + cell_size).intersection(poly)
            if not piece.is_empty:
                sub_pieces = piece.geoms if piece.geom_type == "MultiPolygon" else [piece]
                pieces.extend(g for g in sub_pieces if g.area >= MIN_PARCEL_M2)
            y += cell_size
        x += cell_size
    return pieces


def _lines(fc):
    return [shape(f["geometry"]) for f in fc["features"] if f["geometry"]["type"] == "LineString"]


def _polys(fc):
    return [shape(f["geometry"]) for f in fc["features"] if f["geometry"]["type"] == "Polygon"]


def bounded_voronoi_cells(points_xy, clip_poly):
    """One clipped Voronoi cell per input point, restricted to clip_poly."""
    n = len(points_xy)
    if n == 0:
        return []
    if n == 1:
        return [clip_poly]

    minx, miny, maxx, maxy = clip_poly.bounds
    span = max(maxx - minx, maxy - miny, 1.0)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    far = span * FAR_MULTIPLIER
    mirrors = np.array([
        [cx + far, cy], [cx - far, cy], [cx, cy + far], [cx, cy - far],
        [cx + far, cy + far], [cx + far, cy - far], [cx - far, cy + far], [cx - far, cy - far],
    ])
    all_points = np.vstack([points_xy, mirrors])
    vor = Voronoi(all_points)

    cells = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(clip_poly)
            continue
        verts = [vor.vertices[v] for v in region]
        cell = Polygon(verts)
        if not cell.is_valid:
            cell = cell.buffer(0)
        clipped = cell.intersection(clip_poly)
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        cells.append(clipped)
    return cells


def classify_landuse(parcel_geom, has_building, landuse_polys, govt_union):
    if govt_union is not None and not govt_union.is_empty and parcel_geom.intersects(govt_union):
        return "Government"
    best_label, best_overlap = None, 0.0
    for poly, label in landuse_polys:
        overlap = parcel_geom.intersection(poly).area
        if overlap > best_overlap:
            best_overlap, best_label = overlap, label
    if best_label and best_overlap > 0.3 * parcel_geom.area:
        # an OSM "residential" zoning tag on land with no building yet describes
        # a planning designation, not an individual plot -- showing it the same
        # as an actual house's capped footprint-plot is misleading, so it falls
        # back to vacant. Other zone tags (farmland, forest, industrial...)
        # meaningfully describe the land either way, built or not.
        if best_label == "Residential" and not has_building:
            return "Vacant / Unclassified"
        return best_label
    return "Residential" if has_building else "Vacant / Unclassified"


LANDUSE_LABELS = {
    "farmland": "Agricultural", "farmyard": "Agricultural", "orchard": "Agricultural", "meadow": "Agricultural",
    "forest": "Forest / Green Land", "wood": "Forest / Green Land", "scrub": "Forest / Green Land", "grassland": "Forest / Green Land",
    "industrial": "Industrial", "residential": "Residential", "commercial": "Commercial", "retail": "Commercial",
}


def build_parcels():
    aoi_geo = json.loads((PROC_DIR / "aoi_image_geo.json").read_text(encoding="utf-8"))
    buildings_fc = json.loads((PROC_DIR / "extracted_buildings.geojson").read_text(encoding="utf-8"))
    roads_fc = load_geojson(PROC_DIR / "roads.geojson")
    rail_fc = load_geojson(PROC_DIR / "railway.geojson")
    water_fc = load_geojson(PROC_DIR / "waterway.geojson")
    govt_fc = load_geojson(PROC_DIR / "government.geojson")
    landuse_fc = load_geojson(PROC_DIR / "landuse.geojson") if (PROC_DIR / "landuse.geojson").exists() else {"features": []}
    return build_parcels_from_data(aoi_geo, buildings_fc, roads_fc, rail_fc, water_fc, govt_fc, landuse_fc)


def build_parcels_from_data(aoi_geo, buildings_fc, roads_fc, rail_fc, water_fc, govt_fc, landuse_fc):
    """Same delineation build_parcels() does, but from already-loaded data
    instead of fixed files -- what the upload feature uses for an arbitrary
    new AOI (its own uploaded image's bounds + whatever OSM layers were
    live-fetched for it), so the exact same parcel logic runs on it."""
    lon0 = (aoi_geo["lon_nw"] + aoi_geo["lon_se"]) / 2
    lat0 = (aoi_geo["lat_nw"] + aoi_geo["lat_se"]) / 2
    proj = LocalProjection(lon0, lat0)

    aoi_box = Polygon([
        (aoi_geo["lon_nw"], aoi_geo["lat_se"]), (aoi_geo["lon_se"], aoi_geo["lat_se"]),
        (aoi_geo["lon_se"], aoi_geo["lat_nw"]), (aoi_geo["lon_nw"], aoi_geo["lat_nw"]),
    ])
    aoi_box_local = proj.to_local(aoi_box)

    roads_local = [proj.to_local(g) for g in _lines(roads_fc)]
    rail_local = [proj.to_local(g) for g in _lines(rail_fc)]
    water_lines_local = [proj.to_local(g) for g in _lines(water_fc)]
    water_polys_local = [proj.to_local(g) for g in _polys(water_fc)]
    govt_local = [proj.to_local(g) for g in _polys(govt_fc)]
    govt_union = unary_union(govt_local) if govt_local else None

    landuse_local = []
    for f in landuse_fc["features"]:
        if f["geometry"]["type"] != "Polygon":
            continue
        tag = f["properties"].get("landuse") or f["properties"].get("natural")
        label = LANDUSE_LABELS.get(tag)
        if label:
            landuse_local.append((proj.to_local(shape(f["geometry"])), label))

    infra = [r.buffer(ROAD_BUFFER_M) for r in roads_local] + \
            [r.buffer(RAIL_BUFFER_M) for r in rail_local] + \
            [w.buffer(WATER_BUFFER_M) for w in water_lines_local]
    infra_union = unary_union(infra) if infra else Polygon()
    water_union_local = unary_union(water_polys_local) if water_polys_local else Polygon()
    road_union_local = unary_union(roads_local) if roads_local else None

    land = aoi_box_local.difference(infra_union).difference(water_union_local)
    blocks = list(land.geoms) if land.geom_type == "MultiPolygon" else [land]

    buildings_local = []
    for f in buildings_fc["features"]:
        g = proj.to_local(shape(f["geometry"]))
        buildings_local.append({"id": f["properties"]["id"], "geom": g, "centroid": g.centroid,
                                 "unrecorded": f["properties"].get("unrecorded", False)})

    parcels = []
    pid = 0
    for block in blocks:
        if block.is_empty or block.area < MIN_PARCEL_M2:
            continue
        in_block = [b for b in buildings_local if b["centroid"].within(block)]

        cells_with_members = []
        if in_block:
            pts = np.array([[b["centroid"].x, b["centroid"].y] for b in in_block])
            raw_cells = bounded_voronoi_cells(pts, block)
            for b, raw_cell in zip(in_block, raw_cells):
                # cap the raw Voronoi cell to this building's own footprint + a
                # yard/setback margin, so an isolated building doesn't inherit
                # the whole surrounding empty block as its "plot"
                plot_bound = b["geom"].buffer(PLOT_SETBACK_M, join_style=2)
                capped = raw_cell.intersection(plot_bound)
                if capped.geom_type == "MultiPolygon":
                    capped = max(capped.geoms, key=lambda g: g.area) if capped.geoms else Polygon()
                if not capped.is_empty:
                    cells_with_members.append((capped, [b]))

        # land left over in this block once every building's plot is capped
        # becomes its own separate vacant/unbuilt parcel -- not absorbed into
        # a neighbour's plot
        claimed = unary_union([c for c, _ in cells_with_members]) if cells_with_members else Polygon()
        leftover = block.difference(claimed) if not claimed.is_empty else block
        if not leftover.is_empty:
            leftover_parts = list(leftover.geoms) if leftover.geom_type == "MultiPolygon" else [leftover]
            for part in leftover_parts:
                provisional_landuse = classify_landuse(part, False, landuse_local, govt_union)
                is_tagged_field = provisional_landuse in ("Agricultural", "Forest / Green Land")

                if is_tagged_field:
                    # this patch is REAL, OSM-tagged farmland/forest -- subdivide it
                    # into field-sized parcels, since we have an actual basis for
                    # treating it as several plots, not one
                    if part.area <= FIELD_UNSPLIT_M2:
                        cells_with_members.append((part, []))
                        continue
                    cell_size = math.sqrt(part.area / FIELD_TARGET_CELLS)
                    cell_size = max(FIELD_CELL_MIN_M, min(FIELD_CELL_MAX_M, cell_size))
                    for sub in subdivide_grid(part, cell_size):
                        cells_with_members.append((sub, []))
                elif part.area <= UNTAGGED_VACANT_CAP_M2:
                    # untagged gap/open land -- no evidence it's several separate
                    # plots, so it stays one parcel rather than inventing boundaries
                    cells_with_members.append((part, []))
                else:
                    # only an untagged area this large gets split, purely to avoid
                    # one absurdly oversized parcel -- coarse cells, no plot claim
                    for sub in subdivide_grid(part, UNTAGGED_CELL_M):
                        cells_with_members.append((sub, []))

        for cell, member_buildings in cells_with_members:
            if cell.is_empty or cell.area < MIN_PARCEL_M2:
                continue
            has_building = len(member_buildings) > 0
            landuse = classify_landuse(cell, has_building, landuse_local, govt_union)
            road_dist = cell.distance(road_union_local) if road_union_local is not None else None
            connected = (road_dist is not None) and (road_dist <= ROAD_CONNECT_M)
            any_unrecorded = any(b["unrecorded"] for b in member_buildings)

            centroid_local = cell.centroid
            gps = proj.inv(centroid_local.x, centroid_local.y)
            cell_lonlat = proj.to_lonlat(cell)

            parcels.append({
                "id": pid,
                "geometry": mapping(cell_lonlat),
                "area_m2": round(cell.area, 1),
                "perimeter_m": round(cell.length, 1),
                "landuse": landuse,
                "building_count": len(member_buildings),
                "building_ids": [b["id"] for b in member_buildings],
                "has_unrecorded_building": any_unrecorded,
                "road_connected": bool(connected),
                "road_distance_m": round(road_dist, 1) if road_dist is not None else None,
                "gps_lat": round(gps[1], 6),
                "gps_lon": round(gps[0], 6),
            })
            pid += 1

    return parcels


def main():
    parcels = build_parcels()
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {k: v for k, v in p.items() if k != "geometry"}, "geometry": p["geometry"]}
            for p in parcels
        ],
    }
    out_path = PROC_DIR / "parcels.geojson"
    out_path.write_text(json.dumps(fc), encoding="utf-8")
    n_connected = sum(1 for p in parcels if p["road_connected"])
    print(f"{len(parcels)} parcels delineated -> {out_path}")
    print(f"  {n_connected}/{len(parcels)} have road frontage within {ROAD_CONNECT_M} m")
    from collections import Counter
    print("  land-use mix:", dict(Counter(p["landuse"] for p in parcels)))


if __name__ == "__main__":
    main()
