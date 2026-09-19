"""Build SAM-refined instance labels for dense-area training.

For each tile of a drone mosaic:
  - take the open building footprints that fall in it (Google / Microsoft)
  - prompt SAM with each footprint (box + inside points + outside points)
  - keep the mask that covers >=60% of the footprint and is not 3x bigger
  - snap the outline to the roof's own right angles
Targets written per tile: building mask, shrunk interior (seeds), edge ring,
distance-to-edge, plus road (OSM) and vegetation (excess-green) for the
semantic head.

usage: build_labels.py <out_dir> <n_tiles|all> [tile_m]
"""
import sys, json, warnings, numpy as np, cv2, rasterio, geopandas as gpd, torch
from pathlib import Path
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import box as sbox, Polygon
from shapely.affinity import affine_transform, rotate
from scipy import ndimage as ndi
from segment_anything import sam_model_registry, SamPredictor
warnings.filterwarnings("ignore")

OUT = Path(sys.argv[1]); LIMIT = sys.argv[2]; TILE_M = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
GSD = 0.05
DATA = Path.home()/"Desktop/SIH2026/datasets"
TIF = DATA/"vijayawada/vjwd_singhnagar_2p9cm.tif"
GOOGLE = DATA/"footprints/google_vijayawada_mosaic.geojson"
MS = DATA/"footprints/ms_vijayawada_mosaic.geojson"
OSM = DATA/"osm/vijayawada_ways.geojson"
CKPT = Path.home()/"Desktop/SIH2026/hackathon/cadastraai/models/sam/sam_vit_b_01ec64.pth"
EPSG = 32644
OUT.mkdir(parents=True, exist_ok=True)

fps = [gpd.read_file(GOOGLE).to_crs(EPSG)]
if MS.exists(): fps.append(gpd.read_file(MS).to_crs(EPSG))
fp = gpd.GeoDataFrame(geometry=gpd.pd.concat([f.geometry for f in fps], ignore_index=True), crs=EPSG)
fp = fp[fp.geometry.is_valid & (fp.area > 8)].reset_index(drop=True)
roads = gpd.read_file(OSM).to_crs(EPSG) if OSM.exists() else None
print("footprints:", len(fp), "| road ways:", 0 if roads is None else len(roads))

src = rasterio.open(TIF)
import pyproj
tf = pyproj.Transformer.from_crs(src.crs, EPSG, always_xy=True)
x0, y0 = tf.transform(src.bounds.left, src.bounds.bottom)
x1, y1 = tf.transform(src.bounds.right, src.bounds.top)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
pred = SamPredictor(sam_model_registry["vit_b"](checkpoint=str(CKPT)).to(dev))
print("SAM on", dev)

def snap(poly, gsd=GSD, min_edge_m=0.6, rounds=3):
    mrr = poly.minimum_rotated_rectangle
    if mrr.is_empty or mrr.geom_type != "Polygon": return poly
    c = np.array(mrr.exterior.coords)[:4]
    e, f = c[1]-c[0], c[2]-c[1]
    v = e if np.hypot(*e) >= np.hypot(*f) else f
    ang = np.degrees(np.arctan2(v[1], v[0])); cen = poly.centroid
    p = np.array(rotate(poly, -ang, origin=cen).exterior.coords[:-1], float)
    for _ in range(rounds):
        q = p.copy()
        for i in range(len(p)):
            j = (i+1) % len(p)
            if abs(p[j,0]-p[i,0]) >= abs(p[j,1]-p[i,1]):
                m = (p[i,1]+p[j,1])/2; q[i,1] = q[j,1] = m
            else:
                m = (p[i,0]+p[j,0])/2; q[i,0] = q[j,0] = m
        p = q
    keep = [p[0]]
    for pt in p[1:]:
        if np.hypot(*(pt-keep[-1])) >= min_edge_m/gsd: keep.append(pt)
    if len(keep) < 4: keep = list(p)
    out = Polygon(keep)
    if not out.is_valid: out = out.buffer(0)
    if out.is_empty or out.geom_type != "Polygon": return poly
    return rotate(out, ang, origin=cen).simplify(0.06/gsd)

N = int(TILE_M/GSD)
tiles = []
for tx in np.arange(x0, x1-TILE_M, TILE_M):
    for ty in np.arange(y0, y1-TILE_M, TILE_M):
        g = sbox(tx, ty, tx+TILE_M, ty+TILE_M)
        k = fp.sindex.query(g, predicate="intersects")
        if len(k) >= 6: tiles.append((tx, ty, k))
print("candidate tiles:", len(tiles))
rng = np.random.default_rng(0); rng.shuffle(tiles)
if LIMIT != "all": tiles = tiles[:int(LIMIT)]

kept = 0
for ti, (tx, ty, idx) in enumerate(tiles):
    if (OUT/f"tile_{int(tx)}_{int(ty)}.npz").exists():
        continue
    tr = rasterio.transform.from_origin(tx, ty+TILE_M, GSD, GSD)
    img = np.zeros((3, N, N), np.uint8)
    reproject(rasterio.band(src, [1,2,3]), img, src_transform=src.transform, src_crs=src.crs,
              dst_transform=tr, dst_crs=f"EPSG:{EPSG}", resampling=Resampling.cubic)
    rgb = np.ascontiguousarray(img.transpose(1,2,0))
    if (rgb.max(axis=2) > 12).mean() < 0.97:      # skip tiles with missing imagery
        continue
    to_px = lambda g: affine_transform(g, [1/GSD, 0, 0, -1/GSD, -tx/GSD, (ty+TILE_M)/GSD])
    pred.set_image(rgb)
    inst = np.zeros((N, N), np.int32)
    n_ok = 0
    for j, gi in enumerate(idx):
        gp = to_px(fp.geometry.iloc[gi])
        if gp.is_empty or gp.geom_type != "Polygon": continue
        bx0, by0, bx1, by1 = gp.bounds
        if min(bx1-bx0, by1-by0) < 12: continue
        ours = rasterize([(gp, 1)], out_shape=(N, N), dtype="uint8").astype(bool)
        if ours.sum() < 200: continue
        inner = gp.buffer(-0.6/GSD); inner = gp if inner.is_empty or inner.geom_type != "Polygon" else inner
        pos = [list(inner.representative_point().coords)[0]] + [
            (inner.exterior.interpolate(t, normalized=True).x, inner.exterior.interpolate(t, normalized=True).y)
            for t in (0.2, 0.5, 0.8)]
        ring = gp.buffer(2.5/GSD).exterior
        neg = [(ring.interpolate(t, normalized=True).x, ring.interpolate(t, normalized=True).y) for t in (0.15, 0.5, 0.85)]
        m, sc, _ = pred.predict(point_coords=np.array(pos+neg, np.float32),
                                point_labels=np.array([1]*len(pos)+[0]*len(neg), np.int32),
                                box=np.array([bx0, by0, bx1, by1], np.float32), multimask_output=True)
        best, biou = None, 0.0
        for k in range(m.shape[0]):
            mk = m[k]; inter = (mk & ours).sum()
            if inter == 0: continue
            if inter/ours.sum() < 0.60 or mk.sum() > 3*ours.sum(): continue
            iou = inter/(mk | ours).sum()
            if iou > biou: biou, best = iou, mk
        if best is None: continue
        ctrs, _ = cv2.findContours(best.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ct = max(ctrs, key=cv2.contourArea)
        raw = Polygon(cv2.approxPolyDP(ct, 0.10/GSD, True).reshape(-1, 2))
        if not raw.is_valid: raw = raw.buffer(0)
        if raw.is_empty or raw.geom_type != "Polygon": continue
        mrr = raw.minimum_rotated_rectangle
        fit = mrr if (mrr.area and raw.area/mrr.area > 0.88) else snap(raw)
        n_ok += 1
        inst[rasterize([(fit, 1)], out_shape=(N, N), dtype="uint8") > 0] = n_ok
    if n_ok < 4: continue

    bmask = (inst > 0).astype(np.uint8)
    interior = np.zeros_like(bmask)
    edge = np.zeros_like(bmask)
    for v in range(1, n_ok+1):
        m1 = (inst == v)
        if not m1.any(): continue
        er = ndi.binary_erosion(m1, np.ones((21, 21), bool))       # ~1 m shrink at 5 cm
        interior |= er.astype(np.uint8)
        edge |= (m1 ^ ndi.binary_erosion(m1, np.ones((9, 9), bool))).astype(np.uint8)
    dist = ndi.distance_transform_edt(bmask > 0).astype(np.float32)
    dist = np.clip(dist/ (2.0/GSD), 0, 1)                          # normalised, capped at 2 m
    r = rgb.astype(np.float32)
    exg = (2*r[..., 1] - r[..., 0] - r[..., 2])/255.0              # excess green: vegetation cue
    veg = ((exg > 0.08) & (bmask == 0)).astype(np.uint8)
    road = np.zeros_like(bmask)
    if roads is not None:
        sel = roads.iloc[roads.sindex.query(sbox(tx, ty, tx+TILE_M, ty+TILE_M), predicate="intersects")]
        if len(sel):
            geoms = [to_px(g).buffer(1.6/GSD) for g in sel.geometry]
            road = (rasterize([(g, 1) for g in geoms], out_shape=(N, N), dtype="uint8") > 0).astype(np.uint8)
            road[bmask > 0] = 0
    np.savez_compressed(OUT/f"tile_{int(tx)}_{int(ty)}.npz", rgb=rgb, inst=inst.astype(np.int16),
                        building=bmask, interior=interior, edge=edge, dist=dist, veg=veg, road=road,
                        meta=np.array([tx, ty, TILE_M, GSD], np.float64))
    kept += 1
    if ti % 5 == 0 or kept <= 3:
        print(f"[{ti+1}/{len(tiles)}] tile {int(tx)},{int(ty)}: {len(idx)} footprints -> {n_ok} refined  (kept {kept})")
print("tiles written:", kept, "->", OUT)
