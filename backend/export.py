"""GIS-ready cadastral export: GeoPackage, Shapefile (zipped) and GeoJSON.

GeoPackage and Shapefile carry every layer the AI extracted: parcels (with their
land-cover shares), building footprints, roads & lanes and the road centreline network. GeoJSON is parcels only.

Cadastral layers are delivered in a projected metre CRS so areas and lengths
are true -- the local UTM zone for the AOI (EPSG:32643 for most of
Maharashtra) -- while GeoJSON stays in EPSG:4326 as the format requires.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import shapely
from pyproj import CRS
from shapely.ops import unary_union

EXPORT_FIELDS = ["parcel_id", "prov_pin", "area_m2", "perim_m", "landcover", "built_pct",
                 "road_front", "confidence", "status", "source", "issues", "rec_status", "rec_iou", "rec_ref"]
# share of each land-cover class inside a parcel, % (short names: Shapefile fields are at most 10 characters)
LC_FIELDS = {"building": "lc_bldg", "road": "lc_road", "paved / developed": "lc_paved", "tree": "lc_tree",
             "grass / scrub": "lc_grass", "agriculture": "lc_farm", "bare land": "lc_bare", "water": "lc_water"}


def utm_crs_for(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def parcels_to_gdf(parcels_fc):
    """parcels_fc: GeoJSON FeatureCollection in EPSG:4326."""
    gdf = gpd.GeoDataFrame.from_features(parcels_fc["features"], crs="EPSG:4326")
    # make_valid can return a GeometryCollection with stray lines; keep only area parts
    gdf["geometry"] = gdf.geometry.make_valid().map(
        lambda g: unary_union([p for p in getattr(g, "geoms", [g]) if p.geom_type in ("Polygon", "MultiPolygon")]))
    gdf = gdf[~gdf.geometry.is_empty]
    rename = {"id": "parcel_id", "perimeter_m": "perim_m", "road_frontage": "road_front",
              "record_status": "rec_status", "record_iou": "rec_iou", "record_ref_id": "rec_ref"}
    gdf = gdf.rename(columns=rename)
    for col in EXPORT_FIELDS:
        if col not in gdf.columns:
            gdf[col] = None
    if gdf["issues"].map(lambda v: isinstance(v, list)).any():
        gdf["issues"] = gdf["issues"].map(lambda v: ",".join(v) if isinstance(v, list) else v)
    fields = list(EXPORT_FIELDS)
    if "land_cover_pct" in gdf.columns:
        for cls, col in LC_FIELDS.items():
            gdf[col] = gdf["land_cover_pct"].map(lambda d: float((d or {}).get(cls, 0.0)) if isinstance(d, dict) else None)
        fields += list(LC_FIELDS.values())
    return gdf[fields + ["geometry"]]


def _layer_gdf(fc, rename):
    if not fc or not fc.get("features"):
        return None
    gdf = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326").rename(columns=rename)
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf[~gdf.geometry.is_empty]


def export(parcels_fc, fmt, buildings_fc=None, roads_fc=None, centrelines_fc=None):
    """Returns (bytes, filename, mimetype) for fmt in {gpkg, shp, geojson}."""
    gdf = parcels_to_gdf(parcels_fc)
    if fmt == "geojson":
        return gdf.to_json().encode("utf-8"), "parcels.geojson", "application/geo+json"
    if fmt not in ("gpkg", "shp"):
        raise ValueError(f"Unsupported export format: {fmt}")

    minx, miny, maxx, maxy = gdf.total_bounds
    crs = utm_crs_for((minx + maxx) / 2, (miny + maxy) / 2)
    layers = {"parcels": gdf.to_crs(crs)}
    b = _layer_gdf(buildings_fc, {"id": "bldg_id"})
    if b is not None:
        layers["buildings"] = b.to_crs(crs)[[c for c in ("bldg_id", "area_m2", "source") if c in b.columns] + ["geometry"]]
    r = _layer_gdf(roads_fc, {"id": "road_id", "median_width_m": "med_width", "length_m": "length_m"})
    if r is not None:
        r = r.to_crs(crs)
        r["area_m2"] = r.geometry.area.round(1)
        layers["roads"] = r[[c for c in ("road_id", "med_width", "length_m", "area_m2") if c in r.columns] + ["geometry"]]
    cl = _layer_gdf(centrelines_fc, {"id": "seg_id", "width_m": "width_m", "class": "road_class"})
    if cl is not None:
        layers["road_centrelines"] = cl.to_crs(crs)[["seg_id", "road_class", "width_m", "length_m", "geometry"]]
    with tempfile.TemporaryDirectory() as tmp:
        if fmt == "gpkg":
            path = Path(tmp) / "cadastral.gpkg"
            for name, layer in layers.items():
                layer.to_file(path, layer=name, driver="GPKG")
            return path.read_bytes(), "cadastral.gpkg", "application/geopackage+sqlite3"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, layer in layers.items():
                if name != "road_centrelines":
                    layer = layer.set_geometry(shapely.orient_polygons(layer.geometry.values, exterior_cw=True))
                shp = "road_lines" if name == "road_centrelines" else name   # Shapefile names stay short
                layer.to_file(Path(tmp) / f"{shp}.shp", driver="ESRI Shapefile")
                for f in Path(tmp).glob(f"{shp}.*"):
                    zf.write(f, f.name)
        return buf.getvalue(), "cadastral_shp.zip", "application/zip"
