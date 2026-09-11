"""Small dependency-free geo helpers: lon/lat <-> local metres, and a couple
of GeoJSON I/O convenience functions. Buffers/distances for the rule engine
must happen in metres, not degrees, so every polygon is projected into a
local equirectangular metre grid centred on the AOI before any shapely
buffer/intersection call, then projected back for output.
"""
import json
import math
from pathlib import Path

from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

EARTH_R = 6378137.0  # metres, WGS84 equatorial radius


class LocalProjection:
    """Equirectangular projection centred at (lon0, lat0). Accurate to a few
    centimetres over an AOI a few kilometres wide -- more than sufficient
    for this scale, and avoids a pyproj/GDAL dependency."""

    def __init__(self, lon0, lat0):
        self.lon0 = lon0
        self.lat0 = lat0
        self.k = math.cos(math.radians(lat0))

    def fwd(self, lon, lat):
        x = math.radians(lon - self.lon0) * EARTH_R * self.k
        y = math.radians(lat - self.lat0) * EARTH_R
        return x, y

    def inv(self, x, y):
        lat = math.degrees(y / EARTH_R) + self.lat0
        lon = math.degrees(x / (EARTH_R * self.k)) + self.lon0
        return lon, lat

    def to_local(self, geom):
        return shp_transform(lambda lon, lat: self.fwd(lon, lat), geom)

    def to_lonlat(self, geom):
        return shp_transform(lambda x, y: self.inv(x, y), geom)


def load_geojson(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def features_to_geoms(fc):
    return [shape(f["geometry"]) for f in fc["features"]]


def geoms_to_fc(geoms, props_list=None):
    props_list = props_list or [{} for _ in geoms]
    feats = [{"type": "Feature", "properties": p, "geometry": mapping(g)} for g, p in zip(geoms, props_list)]
    return {"type": "FeatureCollection", "features": feats}
