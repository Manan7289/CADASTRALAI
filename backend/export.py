"""GIS-ready cadastral export: GeoPackage, Shapefile (zipped) and GeoJSON.

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
                 "road_front", "confidence", "status", "source", "issues"]


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
    rename = {"id": "parcel_id", "perimeter_m": "perim_m", "road_frontage": "road_front"}
    gdf = gdf.rename(columns=rename)
    for col in EXPORT_FIELDS:
        if col not in gdf.columns:
            gdf[col] = None
    if gdf["issues"].map(lambda v: isinstance(v, list)).any():
        gdf["issues"] = gdf["issues"].map(lambda v: ",".join(v) if isinstance(v, list) else v)
    return gdf[EXPORT_FIELDS + ["geometry"]]


def export(parcels_fc, fmt):
    """Returns (bytes, filename, mimetype) for fmt in {gpkg, shp, geojson}."""
    gdf = parcels_to_gdf(parcels_fc)
    if fmt == "geojson":
        return gdf.to_json().encode("utf-8"), "parcels.geojson", "application/geo+json"

    minx, miny, maxx, maxy = gdf.total_bounds
    gdf_utm = gdf.to_crs(utm_crs_for((minx + maxx) / 2, (miny + maxy) / 2))
    with tempfile.TemporaryDirectory() as tmp:
        if fmt == "gpkg":
            path = Path(tmp) / "parcels.gpkg"
            gdf_utm.to_file(path, layer="parcels", driver="GPKG")
            return path.read_bytes(), "parcels.gpkg", "application/geopackage+sqlite3"
        if fmt == "shp":
            path = Path(tmp) / "parcels.shp"
            gdf_utm = gdf_utm.set_geometry(shapely.orient_polygons(gdf_utm.geometry.values, exterior_cw=True))
            gdf_utm.to_file(path, driver="ESRI Shapefile")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in Path(tmp).glob("parcels.*"):
                    zf.write(f, f.name)
            return buf.getvalue(), "parcels_shp.zip", "application/zip"
    raise ValueError(f"Unsupported export format: {fmt}")
