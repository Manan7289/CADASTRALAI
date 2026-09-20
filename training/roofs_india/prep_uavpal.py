"""Turn UAVPal (Bhopal drone survey, 2 cm) into roof-training crops at the app's 0.3 m.

UAVPal: Maiti, Oude Elberink, Vosselman, IEEE JSTARS 17 (2024), doi 10.1109/JSTARS.2023.3330758;
data doi 10.17026/dans-z55-6gt4, CC BY-NC-SA 4.0. Hand-drawn classes: building, road, tree,
water, car, background.

The paper warns that close buildings may be merged in the raster labels, and our roof model has to
split touching houses, so first this checks whether the vector annotation keeps one polygon per
house (area spread, how many building polygons share an edge with another). It then builds a 0.3 m
mosaic from the image tiles, rasterises every building polygon as its own instance, and cuts
512 px crops in the same {rgb, inst} format as training/roofs_india: uptrain_* from the west part,
uptest_* from the east strip (spatial hold-out). Also writes a report and overlays to look at.
Runs on CPU with internet; no GPU.
"""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imagecodecs", "tifffile", "geopandas", "pyogrio"], check=False)
import cv2
import numpy as np
import requests

WORK = Path("/kaggle/working"); RAW = Path("/kaggle/tmp/uavpal"); RAW.mkdir(parents=True, exist_ok=True)
OUT = WORK / "uavpal_crops"; OUT.mkdir(exist_ok=True); VIS = WORK / "vis"; VIS.mkdir(exist_ok=True)
HOST = "https://phys-techsciences.datastations.nl"
DOI = "doi:10.17026/DANS-Z55-6GT4"
GSD = 0.30
CROP, STRIDE = 512, 384
TEST_EAST_SHARE = 0.2
report = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(f):
    df = f["dataFile"]; p = RAW / (f.get("directoryLabel") or "") / df["filename"]
    if p.exists() and p.stat().st_size == df["filesize"]:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        try:
            r = requests.get(f"{HOST}/api/access/datafile/{df['id']}", timeout=300); r.raise_for_status()
            p.write_bytes(r.content); return p
        except Exception as e:
            log("retry", df["filename"], e); time.sleep(5 * (i + 1))
    raise RuntimeError(df["filename"])


def read_tile(p):
    """(array, (x0, y0, px_w, px_h)) in EPSG:32643; rasterio if it can decode, else tifffile+imagecodecs."""
    try:
        import rasterio
        with rasterio.open(p) as s:
            a = s.read(); t = s.transform
            return np.moveaxis(a, 0, -1), (t.c, t.f, t.a, t.e)
    except Exception:
        import tifffile
        with tifffile.TiffFile(p) as tf:
            pg = tf.pages[0]; a = pg.asarray()
            tp, sc = pg.tags["ModelTiepointTag"].value, pg.tags["ModelPixelScaleTag"].value
            return (a if a.ndim == 3 else a[..., None]), (tp[3] - tp[0] * sc[0], tp[4] + tp[1] * sc[1], sc[0], -sc[1])


def main():
    meta = requests.get(f"{HOST}/api/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60).json()["data"]["latestVersion"]
    files = meta["files"]
    small = [f for f in files if (f.get("directoryLabel") or "") in ("", "Vector")]
    imgs = [f for f in files if f.get("directoryLabel") == "Image/Tiles"]
    labs = [f for f in files if f.get("directoryLabel") == "Label/Tiles"]
    for f in small:
        fetch(f)
    log(f"files {len(files)} | image tiles {len(imgs)} | label tiles {len(labs)}")

    # ---- 1. the vector annotation: one polygon per house, or merged blocks?
    import geopandas as gpd, pyogrio
    gp = RAW / "Vector" / "Annotation.gpkg"
    layers = pyogrio.list_layers(gp).tolist()
    report["layers"] = [l[0] for l in layers]
    name = next((l[0] for l in layers if "build" in l[0].lower()), None)   # one layer per class
    if name is None:
        log("no building layer; stopping so it can be looked at", layers); save(); return
    g = gpd.read_file(gp, layer=name)
    log("layers", layers, "| building layer", name, len(g), "| columns", list(g.columns), "| crs", g.crs)
    col = name
    b = g.explode(index_parts=False).reset_index(drop=True)
    b = b[b.geometry.area > 1.0].reset_index(drop=True)
    area = b.geometry.area.values
    sidx = b.sindex
    touching = sum(1 for i, geom in enumerate(b.geometry)
                   if any(j != i and geom.buffer(0.05).intersection(b.geometry[j]).length > 1.0 for j in sidx.query(geom.buffer(0.05))))
    report["buildings"] = {"column": col, "polygons": int(len(b)), "area_m2_p10_50_90_99": [round(float(np.percentile(area, q)), 1) for q in (10, 50, 90, 99)],
                           "over_500m2": int((area > 500).sum()), "share_touching_another": round(touching / max(len(b), 1), 3)}
    log("buildings", json.dumps(report["buildings"]))

    # ---- 2. 0.3 m mosaic from the 2 cm tiles (downsampled as they arrive, so memory stays small)
    x0 = y1 = None
    geo = {}
    def one(f):
        p = fetch(f); a, (tx, ty, sx, sy) = read_tile(p)
        k = abs(sx) / GSD
        small_a = cv2.resize(a[..., :3], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        p.unlink()
        return f["dataFile"]["filename"], small_a, (tx, ty)
    with ThreadPoolExecutor(8) as ex:
        tiles = list(ex.map(one, imgs))
    log("image tiles read", len(tiles), tiles[0][1].shape, tiles[0][1].dtype)
    xs = [t[2][0] for t in tiles]; ys = [t[2][1] for t in tiles]
    x0, y1 = min(xs), max(ys)
    W = int(round((max(xs) - x0) / GSD)) + max(t[1].shape[1] for t in tiles)
    H = int(round((y1 - min(ys)) / GSD)) + max(t[1].shape[0] for t in tiles)
    mos = np.zeros((H, W, 3), np.uint8); cov = np.zeros((H, W), bool)
    for _, a, (tx, ty) in tiles:
        c, r = int(round((tx - x0) / GSD)), int(round((y1 - ty) / GSD))
        a8 = a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)
        valid = a8.max(-1) > 0
        sub = mos[r:r + a8.shape[0], c:c + a8.shape[1]]; vv = valid[:sub.shape[0], :sub.shape[1]]
        sub[vv] = a8[:sub.shape[0], :sub.shape[1]][vv]; cov[r:r + a8.shape[0], c:c + a8.shape[1]] |= vv
    report["mosaic_px"] = [H, W]; report["coverage"] = round(float(cov.mean()), 3)
    log("mosaic", H, W, "coverage", report["coverage"])

    # ---- 3. every building polygon as its own instance on the mosaic grid
    inst = np.zeros((H, W), np.int32)
    for i, geom in enumerate(b.geometry.values[np.argsort(-area)], start=1):   # small last, so they stay on top
        for poly in getattr(geom, "geoms", [geom]):
            ring = np.array(poly.exterior.coords)
            pts = np.stack([(ring[:, 0] - x0) / GSD, (y1 - ring[:, 1]) / GSD], 1).round().astype(np.int32)
            cv2.fillPoly(inst, [pts], i)
            for hole in poly.interiors:
                h = np.array(hole.coords); cv2.fillPoly(inst, [np.stack([(h[:, 0] - x0) / GSD, (y1 - h[:, 1]) / GSD], 1).round().astype(np.int32)], 0)
    inst[~cov] = 0

    # ---- 4. crops; the east strip is held out
    split_x = int(W * (1 - TEST_EAST_SHARE))
    n = {"uptrain": 0, "uptest": 0}
    for r in range(0, H - CROP + 1, STRIDE):
        for c in range(0, W - CROP + 1, STRIDE):
            if cov[r:r + CROP, c:c + CROP].mean() < 0.9:
                continue
            if c < split_x < c + CROP:
                continue                         # straddles the split: used by neither side
            kind = "uptest" if c >= split_x else "uptrain"
            sub = inst[r:r + CROP, c:c + CROP]
            ids, new = np.unique(sub[sub > 0]), np.zeros_like(sub, np.int16)
            for k, v in enumerate(ids, start=1):
                new[sub == v] = k
            np.savez_compressed(OUT / f"{kind}_Bhopal_{r}_{c}.npz", rgb=mos[r:r + CROP, c:c + CROP], inst=new)
            n[kind] += 1
    report["crops"] = n
    log("crops", n)

    # ---- 5. pictures to check by eye
    thumb = cv2.resize(mos, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    edge = cv2.resize(((cv2.dilate((inst > 0).astype(np.uint8), np.ones((3, 3))) - (inst > 0)) > 0).astype(np.uint8), thumb.shape[1::-1]) > 0
    thumb[edge] = (255, 255, 0); cv2.line(thumb, (split_x // 4, 0), (split_x // 4, thumb.shape[0]), (255, 0, 0), 2)
    cv2.imwrite(str(VIS / "mosaic_overview.jpg"), cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
    rng = np.random.default_rng(0)
    picks = sorted(OUT.glob("*.npz")); picks = [picks[i] for i in rng.choice(len(picks), min(6, len(picks)), replace=False)]
    for p in picks:
        d = np.load(p); rgb, ins = d["rgb"].copy(), d["inst"]
        col_ = rng.integers(60, 255, (int(ins.max()) + 1, 3)); col_[0] = 0
        ov = np.where(ins[..., None] > 0, (0.5 * rgb + 0.5 * col_[ins]).astype(np.uint8), rgb)
        cv2.imwrite(str(VIS / f"{p.stem}.jpg"), cv2.cvtColor(np.hstack([rgb, ov]), cv2.COLOR_RGB2BGR))
    for f in (RAW / "LICENSE.txt", RAW / "README.md"):
        if f.exists():
            (OUT / f.name).write_bytes(f.read_bytes())
    save(); log("done")


def save():
    (WORK / "uavpal_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
