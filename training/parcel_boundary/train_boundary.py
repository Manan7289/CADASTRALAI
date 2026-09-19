"""Parcel-boundary model: learn where one parcel ends and the next begins, from aerial imagery.

PS: "AI-based image segmentation models for parcel delineation". A parcel boundary is usually
something visible -- a compound wall, a fence, a hedge, the party wall between two row houses,
the edge of a plot's paving or garden -- but not always, so this is learned from real cadastral
parcels rather than hand-written rules.

Training data (open, CC BY 4.0 / CC0, fetched here):
    image    PDOK Luchtfoto RGB, "Actueel_orthoHR" (8 cm aerial orthophoto of the Netherlands),
             requested at 0.3 m so it matches the rest of the CadastraAI pipeline
    labels   Kadaster BRK Kadastrale Kaart, collection "perceel" (the national cadastral parcels),
             drawn as boundary lines 2 px (0.6 m) wide
Dutch cities are dense, with row houses, garden walls and fences -- the same cues as dense Indian
colonies. Groningen is held out entirely as the test city.

Model: U-Net (ResNet34, ImageNet) -> boundary probability. Loss: BCE (boundaries up-weighted) +
Dice. Augmentation: flips / rotations, colour, haze and blur (satellite-like), scale 0.8-1.25.
Test metric: boundary F-score with 1 m tolerance on Groningen (precision: predicted boundary
within 1 m of a true one; recall: true boundary within 1 m of a predicted one).

Also predicts the boundary map for the CadastraAI survey images attached as cadastraai-demo-rgb.
"""
import glob, io, json, math, os, random, time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/tmp/nl"); TMP.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
GSD, SIDE_M = 0.3, 300.0
PX = int(SIDE_M / GSD)                        # 1000
TILES_PER_CITY = int(os.environ.get("TILES_PER_CITY", 36))
STEPS = int(os.environ.get("STEPS", 6000))
CROP = 512
WMS = ("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap"
       "&layers=Actueel_orthoHR&styles=&crs=EPSG:28992&bbox={x0},{y0},{x1},{y1}&width={w}&height={h}&format=image/jpeg")
FEAT = ("https://api.pdok.nl/kadaster/brk-kadastrale-kaart/ogc/v1/collections/perceel/items?"
        "bbox={x0},{y0},{x1},{y1}&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/28992"
        "&crs=http://www.opengis.net/def/crs/EPSG/0/28992&limit=1000&f=json")
# built-up extents (Dutch RD New, EPSG:28992); Groningen is the held-out test city
CITIES = {
    "amsterdam": (116000, 480000, 128000, 492000), "rotterdam": (88000, 432000, 98000, 441000),
    "den_haag": (77000, 451000, 85000, 459000), "utrecht": (132000, 451000, 140000, 460000),
    "eindhoven": (158000, 380000, 165000, 387000), "tilburg": (130000, 394000, 136000, 400000),
    "almere": (141000, 481000, 148000, 488000), "amersfoort": (153000, 460000, 158000, 465000),
    "zwolle": (200000, 500000, 205000, 505000), "nijmegen": (185000, 426000, 190000, 431000),
    "haarlem": (101000, 486000, 106000, 491000), "groningen": (231000, 579000, 236000, 584000),
}
TEST_CITY = "groningen"
S = requests.Session(); S.headers["User-Agent"] = "cadastraai-research (SIH 2026)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, binary=False, tries=4):
    for k in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                return r.content if binary else r.json()
        except Exception:
            pass
        time.sleep(2 + 3 * k)
    return None


def parcels(x0, y0, x1, y1):
    feats, url = [], FEAT.format(x0=x0, y0=y0, x1=x1, y1=y1)
    while url and len(feats) < 5000:
        j = get(url)
        if not j:
            break
        feats += j.get("features", [])
        url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
    return feats


def boundary_mask(feats, x0, y1):
    m = np.zeros((PX, PX), np.uint8)
    s = 1.0 / GSD
    for f in feats:
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        for p in polys:
            for ring in p:
                pts = np.array([[(x - x0) * s, (y1 - y) * s] for x, y in ring], np.float32)
                cv2.polylines(m, [np.round(pts).astype(np.int32)], True, 1, 2)
    return m


def build(city, box, n, seed):
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 4:
        tries += 1
        cx, cy = rng.uniform(box[0] + 150, box[2] - 150), rng.uniform(box[1] + 150, box[3] - 150)
        x0, y0 = int(cx - SIDE_M / 2), int(cy - SIDE_M / 2); x1, y1 = x0 + int(SIDE_M), y0 + int(SIDE_M)
        feats = parcels(x0, y0, x1, y1)
        if len(feats) < 40:                  # water, park or rural: not what we want to learn
            continue
        jpg = get(WMS.format(x0=x0, y0=y0, x1=x1, y1=y1, w=PX, h=PX), binary=True)
        if not jpg:
            continue
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rgb is None or rgb.shape[:2] != (PX, PX) or rgb.std() < 8:
            continue
        m = boundary_mask(feats, x0, y1)
        f = TMP / f"{city}_{x0}_{y0}.npz"
        np.savez_compressed(f, rgb=rgb, b=m)
        out.append(f)
    log(city, len(out), "tiles")
    return out


# ------------------------------------------------------------------ training
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def augment(rgb, b):
    k = random.randint(0, 3)
    rgb, b = np.rot90(rgb, k), np.rot90(b, k)
    if random.random() < 0.5:
        rgb, b = rgb[:, ::-1], b[:, ::-1]
    s = random.uniform(0.8, 1.25)
    if abs(s - 1) > 0.02:
        rgb = cv2.resize(np.ascontiguousarray(rgb), None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        b = cv2.resize(np.ascontiguousarray(b), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    h, w = b.shape
    y, x = random.randint(0, h - CROP), random.randint(0, w - CROP)
    rgb, b = rgb[y:y + CROP, x:x + CROP].astype(np.float32), b[y:y + CROP, x:x + CROP].astype(np.float32)
    # satellite-like: colour shift, haze, softness
    rgb = rgb * np.random.uniform(0.8, 1.2, 3) + np.random.uniform(-20, 20, 3)
    if random.random() < 0.4:
        rgb = rgb * random.uniform(0.7, 0.95) + 255 * random.uniform(0.05, 0.25)
    if random.random() < 0.5:
        rgb = cv2.GaussianBlur(rgb, (0, 0), random.uniform(0.3, 1.2))
    return np.clip(rgb, 0, 255), b


def batch(files, n):
    xs, ys = [], []
    for _ in range(n):
        d = np.load(random.choice(files))
        rgb, b = augment(d["rgb"], d["b"])
        xs.append(((rgb / 255 - MEAN) / STD).transpose(2, 0, 1)); ys.append(b[None])
    return torch.from_numpy(np.stack(xs)).float().to(DEV), torch.from_numpy(np.stack(ys)).float().to(DEV)


@torch.no_grad()
def predict(net, rgb, tile=512, stride=384):
    h, w = rgb.shape[:2]
    im = np.pad(rgb, ((0, max(0, tile - h)), (0, max(0, tile - w)), (0, 0)), mode="reflect")
    H, W = im.shape[:2]
    x = torch.from_numpy(((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]).to(DEV)
    acc = torch.zeros((H, W), device=DEV); cnt = torch.zeros((H, W), device=DEV)
    ys = sorted(set(list(range(0, H - tile + 1, stride)) + [H - tile])); xs = sorted(set(list(range(0, W - tile + 1, stride)) + [W - tile]))
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(net(x[:, :, y:y + tile, xx:xx + tile]).float())[0, 0]
            acc[y:y + tile, xx:xx + tile] += o; cnt[y:y + tile, xx:xx + tile] += 1
    return (acc / cnt)[:h, :w].cpu().numpy()


def bf_score(prob, gt, tol_px):
    from skimage.morphology import skeletonize
    pred = skeletonize(prob > 0.5)
    gts = skeletonize(gt > 0)
    if not pred.any() or not gts.any():
        return 0.0, 0.0, 0.0
    dg = cv2.distanceTransform((~gts).astype(np.uint8), cv2.DIST_L2, 3)
    dp = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 3)
    p = float((dg[pred] <= tol_px).mean()); r = float((dp[gts] <= tol_px).mean())
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    import segmentation_models_pytorch as smp
    tiles = {}
    for i, (c, box) in enumerate(CITIES.items()):
        tiles[c] = build(c, box, TILES_PER_CITY, seed=i)
    train = [f for c, fs in tiles.items() if c != TEST_CITY for f in fs]
    test = tiles[TEST_CITY]
    log(f"train {len(train)} tiles from {len(CITIES) - 1} cities, test {len(test)} ({TEST_CITY})")
    if len(train) < 20 or len(test) < min(5, TILES_PER_CITY):
        raise SystemExit("not enough tiles fetched")
    pos = np.mean([np.load(f)["b"].mean() for f in train[:60]])
    pw = torch.tensor(min(10.0, (1 - pos) / max(pos, 1e-6))).to(DEV)
    log(f"boundary share {pos:.3f}, positive weight {pw.item():.1f}")

    net = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    # OneCycleLR divides by zero on tiny smoke runs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=STEPS, pct_start=0.05) if STEPS >= 100 else None
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    for step in range(STEPS):
        x, y = batch(train, 8)
        with torch.autocast("cuda", dtype=torch.float16):
            o = net(x)
            bce = F.binary_cross_entropy_with_logits(o.float(), y, pos_weight=pw)
            p = torch.sigmoid(o.float())
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
            loss = bce + dice
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched:
            sched.step()
        if step % 250 == 0:
            log(f"step {step}/{STEPS} loss {loss.item():.3f} ({(time.time() - t0) / 60:.1f} min)")
        if time.time() - t0 > 3.3 * 3600:
            log("time cap reached at step", step); break
    net.eval()
    torch.save({"state_dict": net.state_dict(), "arch": "unet_resnet34", "gsd_m": GSD, "mean": MEAN.tolist(), "std": STD.tolist(),
                "trained_on": "PDOK Luchtfoto 0.3 m + Kadaster BRK parcels, 11 Dutch cities"}, WORK / "parcel_boundary_unet.pt")

    # test city
    tol = int(round(1.0 / GSD))
    scores = []
    vis_dir = WORK / "vis"; vis_dir.mkdir(exist_ok=True)
    for k, f in enumerate(test):
        d = np.load(f); prob = predict(net, d["rgb"])
        scores.append(bf_score(prob, d["b"], tol))
        if k < 4:
            ov = d["rgb"].copy(); ov[d["b"] > 0] = (255, 255, 0)
            pr = d["rgb"].copy(); pr[prob > 0.5] = (255, 0, 255)
            cv2.imwrite(str(vis_dir / f"test_{k}.jpg"), cv2.cvtColor(np.hstack([ov, pr]), cv2.COLOR_RGB2BGR)[::2, ::2])
    s = np.array(scores)
    metrics = {"test_city": TEST_CITY, "tiles": len(test), "tolerance_m": 1.0,
               "precision": round(float(s[:, 0].mean()), 3), "recall": round(float(s[:, 1].mean()), 3),
               "f_score": round(float(s[:, 2].mean()), 3), "train_tiles": len(train), "steps": STEPS}
    log("TEST", metrics)
    (WORK / "boundary_metrics.json").write_text(json.dumps(metrics, indent=2))

    # CadastraAI survey images
    out = WORK / "survey_boundaries"; out.mkdir(exist_ok=True)
    for f in sorted(glob.glob("/kaggle/input/**/cadastraai-demo-rgb/**/*.npz", recursive=True)):
        d = np.load(f, allow_pickle=False)
        prob = predict(net, d["rgb"])
        np.savez_compressed(out / Path(f).name, boundary=(prob * 255).astype(np.uint8))
        pr = d["rgb"].copy(); pr[prob > 0.5] = (255, 0, 255)
        cv2.imwrite(str(vis_dir / f"survey_{Path(f).stem}.jpg"), cv2.cvtColor(np.hstack([d["rgb"], pr]), cv2.COLOR_RGB2BGR)[::2, ::2])
        log("survey", Path(f).stem, "boundary share", round(float((prob > 0.5).mean()), 3))
    log("done")


if __name__ == "__main__":
    main()
