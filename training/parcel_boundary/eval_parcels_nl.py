"""How accurate are CadastraAI's parcels? Tested against official cadastral parcels.

India has no open cadastral parcel layer, so the parcel pipeline is scored in Groningen (the
Netherlands), a city none of the models trained on:

    image     PDOK Luchtfoto, requested at 0.3 m (the pipeline's scale)
    buildings BAG building register footprints (so this measures the parcel step, not roof detection)
    roads     our land-cover model on the image, exactly as in the app
    parcels   the app's parcel step, with and without the parcel-boundary model
    truth     Kadaster BRK cadastral parcels

Scores, on cadastral parcels that contain a building (private plots):
    mean IoU          overlap of each true parcel with its best-matching AI parcel
    matched (IoU>=0.5 / 0.75)
    boundary F @1 m   share of AI boundary within 1 m of a true boundary, and vice versa
Runs on the laptop CPU (under 1 GB): python training/parcel_boundary/eval_parcels_nl.py [n_tiles]
"""
import json, random, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
import cv2
import numpy as np
import requests
import torch
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape

import kaggle_dns  # noqa: F401  (resolver fallback)
import import_bundle as ib
import parcel_extract as pe
import segment

GSD, SIDE = 0.3, 300.0
PX = int(SIDE / GSD)
BOX = (231000, 579000, 236000, 584000)          # Groningen
WMS = ("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap&layers=Actueel_orthoHR"
       "&styles=&crs=EPSG:28992&bbox={x0},{y0},{x1},{y1}&width={w}&height={w}&format=image/jpeg")
BRK = ("https://api.pdok.nl/kadaster/brk-kadastrale-kaart/ogc/v1/collections/perceel/items?bbox={x0},{y0},{x1},{y1}"
       "&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/28992&crs=http://www.opengis.net/def/crs/EPSG/0/28992&limit=1000&f=json")
BAG = ("https://service.pdok.nl/lv/bag/wfs/v2_0?service=WFS&version=2.0.0&request=GetFeature&typeNames=bag:pand"
       "&bbox={x0},{y0},{x1},{y1},urn:ogc:def:crs:EPSG::28992&srsName=EPSG:28992&outputFormat=application/json&count=5000")
S = requests.Session(); S.headers["User-Agent"] = "cadastraai-research (SIH 2026)"


def get(url, binary=False):
    for k in range(4):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                return r.content if binary else r.json()
        except Exception:
            time.sleep(2 + 3 * k)
    return None


def feats_all(url):
    out = []
    while url:
        j = get(url)
        if not j:
            break
        out += j.get("features", [])
        url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
    return out


def boundary_net():
    import segmentation_models_pytorch as smp
    ck = torch.load(ROOT / "models/parcel_boundary_v2_india/parcel_boundary_unet_india.pt", map_location="cpu", weights_only=False)
    net = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    net.load_state_dict(ck["state_dict"])
    return net.eval()


@torch.no_grad()
def boundary(net, rgb, tile=512, stride=384):
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    h, w = rgb.shape[:2]
    x = torch.from_numpy(((rgb.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None])
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    for y in sorted(set(list(range(0, h - tile + 1, stride)) + [h - tile])):
        for xx in sorted(set(list(range(0, w - tile + 1, stride)) + [w - tile])):
            acc[y:y + tile, xx:xx + tile] += torch.sigmoid(net(x[:, :, y:y + tile, xx:xx + tile]))[0, 0].numpy()
            cnt[y:y + tile, xx:xx + tile] += 1
    return acc / cnt


def lines(polys, t):
    m = np.zeros((PX, PX), np.uint8)
    inv = ~t
    for g in polys:
        for p in getattr(g, "geoms", [g]):
            for ring in [p.exterior] + list(p.interiors):
                pts = np.array([inv * c for c in ring.coords], np.float32)
                cv2.polylines(m, [np.round(pts).astype(np.int32)], True, 1, 1)
    return m > 0


def score(ai_polys, gt_polys, houses, t, inner):
    from shapely.strtree import STRtree
    tree = STRtree(ai_polys)
    private = [g for g in gt_polys if any(g.contains(h.representative_point()) for h in houses) and inner.contains(g.representative_point())]
    ious = []
    for g in private:
        best = 0.0
        for j in tree.query(g, predicate="intersects"):
            a = ai_polys[int(j)]
            inter = g.intersection(a).area
            best = max(best, inter / max(g.union(a).area, 1e-6))
        ious.append(best)
    # boundary F-score within the tile interior (tile edges are not parcel boundaries)
    mask = np.zeros((PX, PX), bool)
    b = int(6 / GSD); mask[b:-b, b:-b] = True
    ai_l, gt_l = lines(ai_polys, t) & mask, lines(gt_polys, t) & mask
    tol = round(1.0 / GSD)
    dg = cv2.distanceTransform((~gt_l).astype(np.uint8), cv2.DIST_L2, 3)
    da = cv2.distanceTransform((~ai_l).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((dg[ai_l] <= tol).mean()) if ai_l.any() else 0.0
    r = float((da[gt_l] <= tol).mean()) if gt_l.any() else 0.0
    return ious, p, r


def main(n):
    rng = random.Random(7)
    net = boundary_net()
    lc_path = segment.MODELS_DIR / segment.MODELS["landcover_v2"]["file"]
    rows = {"nearest house": [], "nearest house + boundary model": []}
    for k in range(n * 3):
        if sum(1 for _ in rows["nearest house"]) >= n:
            break
        cx, cy = rng.uniform(BOX[0] + 150, BOX[2] - 150), rng.uniform(BOX[1] + 150, BOX[3] - 150)
        x0, y0 = int(cx - SIDE / 2), int(cy - SIDE / 2); x1, y1 = x0 + int(SIDE), y0 + int(SIDE)
        gt = [shape(f["geometry"]) for f in feats_all(BRK.format(x0=x0, y0=y0, x1=x1, y1=y1)) if f.get("geometry")]
        bag = [shape(f["geometry"]) for f in (get(BAG.format(x0=x0, y0=y0, x1=x1, y1=y1)) or {}).get("features", []) if f.get("geometry")]
        if len(gt) < 40 or len(bag) < 20:
            continue
        jpg = get(WMS.format(x0=x0, y0=y0, x1=x1, y1=y1, w=PX), binary=True)
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        t = Affine(GSD, 0, x0, 0, -GSD, y1)
        houses = rasterize([(g, i + 1) for i, g in enumerate(bag)], out_shape=(PX, PX), transform=t, dtype="int32")
        lcp = segment.predict_landcover(rgb, lc_path)
        p5 = np.zeros((5, PX, PX), np.float32)
        for c, a in ib.LC_TO_APP.items():
            p5[a] += lcp[c]
        lc = (lcp[1:].argmax(0) + 1).astype(np.uint8)
        bnd = boundary(net, rgb)
        from shapely.geometry import box
        inner = box(x0 + 20, y0 + 20, x1 - 20, y1 - 20)
        for name, b in (("nearest house", None), ("nearest house + boundary model", bnd)):
            ex = pe.extract(p5, rgb, t, valid=np.ones((PX, PX), bool), inst_override=houses, landcover=lc, paved=lc == 3,
                            boundary=b)
            ious, p, r = score([f["geometry"] for f in ex["parcels"]], gt, bag, t, inner)
            rows[name].append((ious, p, r))
        print(f"tile {len(rows['nearest house'])}/{n} at {x0},{y0}: " + " | ".join(
            f"{nm}: IoU {np.mean(v[-1][0]):.2f} F {2 * v[-1][1] * v[-1][2] / max(v[-1][1] + v[-1][2], 1e-6):.2f}" for nm, v in rows.items()), flush=True)
    out = {}
    for nm, v in rows.items():
        ious = np.concatenate([np.array(x[0]) for x in v]); p = np.mean([x[1] for x in v]); r = np.mean([x[2] for x in v])
        out[nm] = {"tiles": len(v), "private_parcels": int(ious.size), "mean_iou": round(float(ious.mean()), 3),
                   "matched_iou50_pct": round(100 * float((ious >= 0.5).mean()), 1), "matched_iou75_pct": round(100 * float((ious >= 0.75).mean()), 1),
                   "boundary_precision_1m": round(float(p), 3), "boundary_recall_1m": round(float(r), 3),
                   "boundary_f_1m": round(float(2 * p * r / max(p + r, 1e-6)), 3)}
    print(json.dumps(out, indent=2))
    if len(sys.argv) <= 3:
        (ROOT / "models/parcel_boundary_v1/groningen_parcel_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    import plot_layout
    if len(sys.argv) > 3:          # optional weights sweep: n_tiles boundary_weight distance_weight
        plot_layout.BOUNDARY_WEIGHT, plot_layout.DISTANCE_WEIGHT = float(sys.argv[2]), float(sys.argv[3])
        print("weights", plot_layout.BOUNDARY_WEIGHT, plot_layout.DISTANCE_WEIGHT, flush=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
