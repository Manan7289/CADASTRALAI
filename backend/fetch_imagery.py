"""Download and stitch real satellite/aerial tiles covering the demo AOI into
one georeferenced image. Uses standard OSM "slippy map" tile math (no GDAL
required) so any pixel in the stitched image can be converted back to
lon/lat exactly.

Tile source: Esri World Imagery (public tile service, no API key required
for light/dev use: https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery)
"""
import io
import json
import math
from pathlib import Path

import requests
from PIL import Image

HEADERS = {"User-Agent": "CadastraAI-SIH2026-Demo/1.0 (educational hackathon project)"}
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

BBOX = (19.691, 73.5545, 19.701, 73.567)  # south, west, north, east -- must match fetch_data.py
ZOOM = 17

# Zoom 17 is ~1.1 m/px here, which is fine for the map backdrop and the
# RandomForest baseline. The Inria-trained U-Net (ml/) instead expects ~0.3 m/px,
# the resolution it was trained at, so it fetches zoom 19 (~0.28 m/px) into a
# separate file rather than replacing the backdrop -- a 5120x5120 PNG is far too
# heavy to ship to the browser.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 2 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def fetch_tile(z, x, y, cache_dir):
    cache_path = cache_dir / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    url = TILE_URL.format(z=z, y=y, x=x)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache_dir.mkdir(parents=True, exist_ok=True)
    img.save(cache_path)
    return img


def main(zoom=ZOOM, prefix="aoi_image"):
    south, west, north, east = BBOX
    x0f, y0f = deg2num(north, west, zoom)   # top-left (NW) fractional tile coords
    x1f, y1f = deg2num(south, east, zoom)   # bottom-right (SE) fractional tile coords

    x0, x1 = int(math.floor(x0f)), int(math.floor(x1f))
    y0, y1 = int(math.floor(y0f)), int(math.floor(y1f))

    tiles_x = list(range(x0, x1 + 1))
    tiles_y = list(range(y0, y1 + 1))
    print(f"Downloading {len(tiles_x)} x {len(tiles_y)} = {len(tiles_x)*len(tiles_y)} tiles at zoom {zoom} ...")

    cache_dir = RAW_DIR / "tiles"
    mosaic = Image.new("RGB", (256 * len(tiles_x), 256 * len(tiles_y)))

    for j, ty in enumerate(tiles_y):
        for i, tx in enumerate(tiles_x):
            tile = fetch_tile(zoom, tx, ty, cache_dir)
            mosaic.paste(tile, (i * 256, j * 256))
        print(f"  row {j+1}/{len(tiles_y)} done")

    # geo-bounds of the full mosaic (NW corner of top-left tile, SE corner of bottom-right tile)
    lat_nw, lon_nw = num2deg(x0, y0, zoom)
    lat_se, lon_se = num2deg(x1 + 1, y1 + 1, zoom)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    img_path = PROC_DIR / f"{prefix}.png"
    mosaic.save(img_path)

    geotransform = {
        "width": mosaic.width,
        "height": mosaic.height,
        "lon_nw": lon_nw, "lat_nw": lat_nw,
        "lon_se": lon_se, "lat_se": lat_se,
        "zoom": zoom,
        "source": "Esri World Imagery (public tile service)",
    }
    (PROC_DIR / f"{prefix}_geo.json").write_text(json.dumps(geotransform, indent=2), encoding="utf-8")
    print(f"Saved mosaic {mosaic.width}x{mosaic.height} -> {img_path}")
    print(f"Geo bounds: NW=({lat_nw:.6f},{lon_nw:.6f}) SE=({lat_se:.6f},{lon_se:.6f})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=ZOOM)
    ap.add_argument("--prefix", default="aoi_image")
    a = ap.parse_args()
    main(a.zoom, a.prefix)
