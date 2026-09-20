"""Data for the house-separation ensemble (Kaggle CPU kernel, internet on).

1. RAMP chips (DevGlobal, CC BY-NC 4.0) for South Asian cities: 256 px, ~30 cm Maxar
   imagery, one polygon per building, touching houses drawn as separate polygons.
   Labels are lon/lat GeoJSON; they are reprojected to each chip's CRS and burnt in
   as an instance raster.
2. Geographic split per city: the eastern TEST_SHARE of chips (by centroid longitude)
   is the test area; chips within a BUFFER_M strip west of it are dropped, so no test
   house also sits in a training chip.
3. Nacala-Roof-Material trained weights (CC0): YOLOv8-seg and U-Net-DOW, used as
   starting points.
4. A preview of label overlays per city, to check alignment by eye before training.

Outputs (/kaggle/working):
    ramp/<city>_<split>.npz   rgb (N,256,256,3) uint8, inst (N,256,256) int16
    nacala/yolo1/..., nacala/unet_dow1/...
    ramp_preview.jpg, prep_summary.json
"""
import io
import json
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom

OUT = Path("/kaggle/working")
BASE = "https://data.source.coop/ramp"
CITIES = ["karnataka_india", "dhaka_bangladesh", "sylhet_bangladesh", "chittagong_bangladesh"]
MAX_PER_CITY = 7000
TEST_SHARE = 0.10
BUFFER_M = 250
NACALA = "https://sid.erda.dk/share_redirect/HF2srDrYEa/{}.zip"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, tries=4):
    for t in range(tries):
        try:
            # the Source Cooperative proxy answers 403 to urllib's default user agent
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.4.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # transient network errors on a 20k-file pull
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{BASE}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token, safe="")
        xml = get(url).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def load_chip(city, cid):
    try:
        tif = get(f"{BASE}/ramp/ramp_{city}/source/{cid}.tif")
        gj = json.loads(get(f"{BASE}/ramp/ramp_{city}/labels/{cid}.geojson"))
        with rasterio.MemoryFile(tif) as mf, mf.open() as src:
            if src.count < 3 or src.shape != (256, 256):
                return None
            rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
            shapes = [(transform_geom("EPSG:4326", src.crs, f["geometry"]), i + 1)
                      for i, f in enumerate(gj["features"]) if f.get("geometry")]
            inst = (rasterize(shapes, out_shape=(256, 256), transform=src.transform, dtype="int32")
                    if shapes else np.zeros((256, 256), np.int32))
            # centroid in lon/lat and metres-per-degree, for the geographic split
            b = src.bounds
            cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
            if src.crs.to_epsg() != 4326:
                g = transform_geom(src.crs, "EPSG:4326", {"type": "Point", "coordinates": [cx, cy]})
                cx, cy = g["coordinates"]
        if (rgb.max(-1) == 0).mean() > 0.2:      # mostly outside the image
            return None
        return cid, rgb.astype(np.uint8), inst.astype(np.int16), cx, cy
    except Exception as e:
        return ("error", cid, repr(e)[:120])


def prep_city(city):
    keys = list_keys(f"ramp/ramp_{city}/labels/")
    ids = sorted(Path(k).stem for k in keys if k.endswith(".geojson"))
    rng = np.random.default_rng(0)
    if len(ids) > MAX_PER_CITY:
        ids = list(rng.choice(ids, MAX_PER_CITY, replace=False))
    log(f"{city}: {len(keys)} label files, fetching {len(ids)}")
    t0 = time.time()
    with ThreadPoolExecutor(48) as ex:
        res = list(ex.map(lambda c: load_chip(city, c), ids))
    errs = [r for r in res if r and r[0] == "error"]
    ok = [r for r in res if r and r[0] != "error"]
    log(f"{city}: {len(ok)} chips ok, {len(errs)} errors, {time.time() - t0:.0f}s", errs[:2])
    lon = np.array([r[3] for r in ok]); lat = np.array([r[4] for r in ok])
    m_per_deg = 111_320 * np.cos(np.radians(lat.mean()))
    cut = np.quantile(lon, 1 - TEST_SHARE)
    test = lon >= cut
    buffer = (~test) & (lon >= cut - BUFFER_M / m_per_deg)
    train = ~test & ~buffer
    summary = {}
    for split, sel in (("train", train), ("test", test)):
        idx = np.nonzero(sel)[0]
        rgb = np.stack([ok[i][1] for i in idx]); inst = np.stack([ok[i][2] for i in idx])
        np.savez_compressed(OUT / "ramp" / f"{city}_{split}.npz", rgb=rgb, inst=inst)
        n_b = int(sum(len(np.unique(x)) - 1 for x in inst))
        summary[split] = {"chips": int(len(idx)), "buildings": n_b}
    summary["dropped_in_buffer"] = int(buffer.sum())
    log(city, summary)
    return summary, [ok[i] for i in np.nonzero(train)[0][:6]]


def preview(samples):
    from PIL import Image
    rows = []
    for city, chips in samples.items():
        tiles = []
        for _, rgb, inst, _, _ in chips:
            rng = np.random.default_rng(1)
            cols = (rng.random((int(inst.max()) + 1, 3)) * 255).astype(np.uint8); cols[0] = 0
            ov = rgb.copy(); m = inst > 0
            ov[m] = (0.5 * ov[m] + 0.5 * cols[inst[m]]).astype(np.uint8)
            # outline each building so touching neighbours are visibly separate
            edge = np.zeros_like(m)
            edge[1:] |= inst[1:] != inst[:-1]; edge[:, 1:] |= inst[:, 1:] != inst[:, :-1]
            ov[edge & m] = 255
            tiles.append(np.hstack([rgb, ov]))
        rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(OUT / "ramp_preview.jpg", quality=85)


def fetch_nacala(name):
    t0 = time.time()
    data = get(NACALA.format(name))
    zipfile.ZipFile(io.BytesIO(data)).extractall(OUT / "nacala" / name)
    log(f"nacala {name}: {len(data) / 1e6:.0f} MB in {time.time() - t0:.0f}s ->",
        sorted(str(p.relative_to(OUT)) for p in (OUT / "nacala" / name).rglob("*") if p.is_file())[:10])


def main():
    (OUT / "ramp").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(2) as ex:
        nac = [ex.submit(fetch_nacala, n) for n in ("yolo1", "unet_dow1")]
        summary, samples = {}, {}
        for city in CITIES:
            summary[city], samples[city] = prep_city(city)
        for f in nac:
            f.result()
    preview(samples)
    (OUT / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    log("done", json.dumps(summary))


if __name__ == "__main__":
    main()
