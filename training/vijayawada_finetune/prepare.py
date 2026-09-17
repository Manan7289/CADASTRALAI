"""Build the Indian fine-tuning set: Vijayawada (Singh Nagar) drone orthoimage
tiles at 10 cm with weak building labels from Microsoft Global ML Building
Footprints.

Footprints are traced from satellite imagery, so they are coarse and can be
offset by 1-2 m from the drone image. OpenStreetMap roads/lanes add real
negatives -- without them, narrow lanes between footprints fall entirely in
the ignore band and a model can learn to paint everything as building.
Labels:
  1   building      -- inside a footprint, eroded by INNER_M
  2   road / lane   -- OSM highway centreline buffered by a per-type half-width
  0   not building  -- outside footprints dilated by OUTER_M (includes OSM rail/water)
  255 ignore        -- uncertain band around footprint edges, road/footprint conflicts, nodata

The demo block (plus HOLDOUT_BUFFER_M) is never used for training: it is
written separately as the held-out evaluation tile.

Reads the mosaic window by window (WarpedVRT), so memory stays small.

Several footprint layers can be given (comma-separated); they are combined,
so a building present in either source counts -- open layers each miss
different buildings.

Usage: python prepare.py <mosaic.tif> <footprints.geojson[,more.geojson]> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
import segment  # noqa: E402

GSD = 0.10
TILE = 1024
INNER_M = 0.4
OUTER_M = 0.6
ROAD_HALF_WIDTH_M = {"primary": 5.0, "primary_link": 3.5, "secondary": 4.5, "tertiary": 3.5, "tertiary_link": 2.5,
                     "residential": 1.8, "living_street": 1.5, "service": 1.5, "pedestrian": 1.5, "busway": 3.0,
                     "track": 1.2, "path": 0.8, "footway": 0.8, "unclassified": 2.0}
ROAD_EDGE_MARGIN_M = 0.4   # stay inside the carriageway: OSM centrelines can be offset a little
MIN_VALID = 0.6
DEMO_BLOCK = (80.63652, 16.52445, 80.63885, 16.52678)   # west, south, east, north
HOLDOUT_BUFFER_M = 30


def labels_for(geoms, transform, shape_hw, valid, roads=()):
    inside = rasterize([(g, 1) for g in geoms], out_shape=shape_hw, transform=transform).astype(bool) if geoms else np.zeros(shape_hw, bool)
    r_in, r_out = max(1, int(INNER_M / GSD)), max(1, int(OUTER_M / GSD))
    core = ndi.binary_erosion(inside, iterations=r_in)
    near = ndi.binary_dilation(inside, iterations=r_out)
    lab = np.full(shape_hw, 255, np.uint8)
    lab[core] = 1
    lab[~near] = 0
    road_shapes = [(g.buffer(max(0.3, w - ROAD_EDGE_MARGIN_M), cap_style=2), 1) for g, w in roads]
    if road_shapes:
        road = rasterize(road_shapes, out_shape=shape_hw, transform=transform).astype(bool)
        lab[road & ~inside] = 2
        lab[road & inside] = 255      # footprint and road disagree here: don't teach either
    lab[~valid] = 255
    return lab


def load_ways(path, to_utm):
    """OSM ways -> (utm line, half-width) for roads, and utm lines to treat as not-building for rail/water."""
    roads, other = [], []
    if not path:
        return roads, other
    for f in json.loads(Path(path).read_text())["features"]:
        kind = f["properties"]["kind"]
        g = shp_transform(to_utm.transform, shape(f["geometry"]))
        if kind in ROAD_HALF_WIDTH_M:
            roads.append((g, ROAD_HALF_WIDTH_M[kind]))
        elif kind.startswith(("railway:rail", "waterway:")):
            other.append(g)
    return roads, other


def main(mosaic, footprints, out_dir, ways=None):
    out = Path(out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "holdout").mkdir(parents=True, exist_ok=True)

    with rasterio.open(mosaic) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    crs = segment.utm_crs_for((w + e) / 2, (s + n) / 2)
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, w, s, e, n)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    geoms = []
    for fp in footprints.split(","):
        fc = json.loads(Path(fp).read_text())
        geoms += [shp_transform(to_utm.transform, shape(f["geometry"])) for f in fc["features"]]
    print(f"{len(geoms)} footprints from {len(footprints.split(','))} source(s)")
    roads, _ = load_ways(ways, to_utm)
    print(f"{len(roads)} OSM road/lane centrelines")
    demo = shp_transform(to_utm.transform, box(*DEMO_BLOCK))
    exclusion = demo.buffer(HOLDOUT_BUFFER_M)

    size_m = TILE * GSD
    kept, skipped_valid, skipped_holdout = 0, 0, 0
    y = top
    while y - size_m > bottom:
        x = left
        while x + size_m < right:
            tb = box(x, y - size_m, x + size_m, y)
            if tb.intersects(exclusion):
                skipped_holdout += 1
                x += size_m
                continue
            t = from_origin(x, y, GSD, GSD)
            rgb = segment.warp_to_grid(mosaic, crs, t, TILE, TILE, bands=[1, 2, 3])
            valid = np.isfinite(rgb).all(0) & (np.nan_to_num(rgb).sum(0) > 0)
            if valid.mean() < MIN_VALID:
                skipped_valid += 1
                x += size_m
                continue
            local = [g for g in geoms if g.intersects(tb)]
            local_roads = [(g, w) for g, w in roads if g.intersects(tb.buffer(10))]
            lab = labels_for(local, t, (TILE, TILE), valid, local_roads)
            name = f"vj_{int(x)}_{int(y)}"
            Image.fromarray(np.nan_to_num(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)).save(
                out / "train" / f"{name}.jpg", quality=92)
            Image.fromarray(lab).save(out / "train" / f"{name}_label.png")
            kept += 1
            x += size_m
        y -= size_m

    # held-out demo block, same labelling, never trained on
    l, b, r, tp = demo.bounds
    wpx, hpx = int(np.ceil((r - l) / GSD)), int(np.ceil((tp - b) / GSD))
    t = from_origin(l, tp, GSD, GSD)
    rgb = segment.warp_to_grid(mosaic, crs, t, wpx, hpx, bands=[1, 2, 3])
    valid = np.isfinite(rgb).all(0) & (np.nan_to_num(rgb).sum(0) > 0)
    lab = labels_for([g for g in geoms if g.intersects(demo)], t, (hpx, wpx), valid,
                     [(g, w) for g, w in roads if g.intersects(demo.buffer(10))])
    Image.fromarray(np.nan_to_num(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)).save(out / "holdout" / "demo_block.jpg", quality=95)
    Image.fromarray(lab).save(out / "holdout" / "demo_block_label.png")

    meta = {"gsd_m": GSD, "tile_px": TILE, "crs": crs.to_string(), "train_tiles": kept,
            "skipped_low_valid": skipped_valid, "skipped_holdout": skipped_holdout,
            "label_values": {"0": "not building", "1": "building", "2": "road / lane (OSM)", "255": "ignore"},
            "osm_ways": Path(ways).name if ways else None,
            "inner_m": INNER_M, "outer_m": OUTER_M, "holdout": {"bbox_lonlat": DEMO_BLOCK, "buffer_m": HOLDOUT_BUFFER_M},
            "footprints": [Path(fp).name for fp in footprints.split(",")],
            "imagery": "OpenAerialMap VJWD_FLOODS, Singh Nagar, Vijayawada, 2.9 cm drone ORI (Bhuvan)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])
