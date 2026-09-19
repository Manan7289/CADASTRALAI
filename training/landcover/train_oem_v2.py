"""Land-cover v2 (roads, vegetation, bare land focus). Changes from v1: trained to the end
(no 70-minute cap), extra loss weight on bare land and roads, crops centred on bare
land / road half the time, zoom augmentation for 0.25-0.5 m imagery, and a choice of
architecture: ARCH=unet_r50 (as v1) or ARCH=segformer_b2 (transformer encoder with a
wider view, which suits long thin roads).

Land-cover model for the PS's land-use layer: roads, trees, grass/scrub, bare land,
water, farmland, paved areas, buildings. Trained on OpenEarthMap (0.25-0.5 m, 44
countries), scored on its validation split, then combined with the frozen roofs v1
model on 8 Indian city layouts (Esri World Imagery, zoom 19).

Label 0 in OpenEarthMap is "unlabelled" and is ignored in the loss and the scores.
"""
import glob, io, json, math, os, random, sys, time
from pathlib import Path
import numpy as np, cv2, requests, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp

NAMES = ["unlabelled", "bare land", "rangeland (grass/scrub)", "developed space (paved)", "road", "tree",
         "water", "agriculture", "building"]
COLS = np.array([[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148], [255, 255, 255], [34, 97, 38],
                 [0, 69, 255], [75, 181, 73], [222, 31, 7]], np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP, DEV, WORK = 512, "cuda", Path("/kaggle/working")
EPOCHS, TIME_CAP_MIN = int(os.environ.get("EPOCHS", 40)), 330
ARCH = os.environ.get("ARCH", "unet_r50")
random.seed(0); np.random.seed(0); torch.manual_seed(0)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def pairs(split):
    labs = {Path(p).name: p for p in glob.glob(f"/kaggle/input/**/label/{split}/*.tif", recursive=True)}
    return [(p, labs[Path(p).name]) for p in sorted(glob.glob(f"/kaggle/input/**/images/{split}/*.tif", recursive=True))
            if Path(p).name in labs]


def load(ip, lp):
    im = np.array(Image.open(ip).convert("RGB")); lb = np.array(Image.open(lp)).astype(np.int64)
    if lb.shape != im.shape[:2]:
        lb = cv2.resize(lb.astype(np.uint8), im.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return im, np.clip(lb, 0, 8)


class OEM(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train
    def __len__(self):
        return len(self.items) * (2 if self.train else 1)
    def __getitem__(self, i):
        im, lb = load(*self.items[i % len(self.items)])
        h, w = lb.shape
        if h < CROP or w < CROP:
            ph, pw = max(0, CROP - h), max(0, CROP - w)
            im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect"); lb = np.pad(lb, ((0, ph), (0, pw)))
            h, w = lb.shape
        if self.train and np.random.rand() < 0.6:
            s = np.random.uniform(0.7, 1.4)          # imagery comes at 0.25-0.5 m: train across zooms
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            lb = cv2.resize(lb.astype(np.uint8), im.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(np.int64)
            h, w = lb.shape
            if h < CROP or w < CROP:
                ph, pw = max(0, CROP - h), max(0, CROP - w)
                im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="reflect"); lb = np.pad(lb, ((0, ph), (0, pw)))
                h, w = lb.shape
        if self.train:
            y, x = np.random.randint(0, h - CROP + 1), np.random.randint(0, w - CROP + 1)
            if np.random.rand() < 0.5:
                # centre half the crops on the rare classes the PS cares about: bare land and roads
                ys, xs = np.nonzero((lb == 1) | (lb == 4))
                if len(ys):
                    k = np.random.randint(len(ys))
                    y = int(np.clip(ys[k] - CROP // 2, 0, h - CROP)); x = int(np.clip(xs[k] - CROP // 2, 0, w - CROP))
        else:
            y, x = (h - CROP) // 2, (w - CROP) // 2
        im, lb = im[y:y + CROP, x:x + CROP], lb[y:y + CROP, x:x + CROP]
        if self.train:
            k = np.random.randint(4); im, lb = np.rot90(im, k), np.rot90(lb, k)
            if np.random.rand() < 0.5: im, lb = im[:, ::-1], lb[:, ::-1]
            im = np.clip(im.astype(np.float32) * np.random.uniform(0.85, 1.15) + np.random.uniform(-12, 12), 0, 255)
        x4 = ((np.ascontiguousarray(im).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(x4.astype(np.float32)), torch.from_numpy(np.ascontiguousarray(lb))


class FullRes(torch.nn.Module):
    """some decoders (SegFormer) predict at 1/4 resolution: upsample to the input size"""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(x)
        return o if o.shape[-2:] == x.shape[-2:] else F.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice(logits, y, n=9):
    p = logits.softmax(1); m = (y > 0).unsqueeze(1).float()
    oh = F.one_hot(y.clamp_min(0), n).permute(0, 3, 1, 2).float()
    inter = (p * oh * m).sum((0, 2, 3)); den = ((p + oh) * m).sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (den + 1))[1:].mean()


@torch.no_grad()
def predict(model, rgb, tile=512, stride=384):
    """sliding window with overlap; returns per-pixel class 1..8"""
    h, w = rgb.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    im = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = im.shape[:2]
    x = torch.from_numpy(((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]).to(DEV)
    acc = torch.zeros((1, 9, H, W), device=DEV); cnt = torch.zeros((1, 1, H, W), device=DEV)
    ys = list(range(0, H - tile + 1, stride)) + ([H - tile] if (H - tile) % stride else [])
    xs = list(range(0, W - tile + 1, stride)) + ([W - tile] if (W - tile) % stride else [])
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x[:, :, y:y + tile, xx:xx + tile]).float().softmax(1)
            acc[:, :, y:y + tile, xx:xx + tile] += o; cnt[:, :, y:y + tile, xx:xx + tile] += 1
    p = (acc / cnt)[0, 1:].argmax(0) + 1
    return p[:h, :w].cpu().numpy()


def score(model, items):
    conf = np.zeros((9, 9), np.int64); per_region = {}
    for ip, lp in items:
        im, lb = load(ip, lp)
        pr = predict(model, im)
        m = lb > 0
        c = np.bincount(lb[m] * 9 + pr[m], minlength=81).reshape(9, 9)
        conf += c
        reg = Path(ip).stem.rsplit("_", 1)[0]
        per_region[reg] = per_region.get(reg, np.zeros((9, 9), np.int64)) + c
    def ious(cf):
        inter = np.diag(cf)[1:]; union = cf[1:].sum(1) + cf[:, 1:].sum(0)[:] * 0 + cf.sum(0)[1:] - inter
        return inter / np.maximum(union, 1)
    iou = ious(conf)
    res = {"mIoU": float(iou.mean()), "pixel_accuracy": float(np.diag(conf)[1:].sum() / max(1, conf[1:].sum())),
           "per_class_IoU": {NAMES[k + 1]: round(float(v), 3) for k, v in enumerate(iou)}}
    res["south_asia_mIoU"] = {r: round(float(ious(c)[ious(c) > 0].mean()), 3) for r, c in per_region.items()
                              if r in ("dhaka", "coxsbazar", "lohur")}
    return res


def main():
    tr, va = pairs("train"), pairs("val")
    log(f"train {len(tr)} | val {len(va)}")
    if len(tr) < 1000 or len(va) < 100:
        raise SystemExit("missing inputs")
    if ARCH == "segformer_b2":
        model = smp.Segformer("mit_b2", encoder_weights="imagenet", in_channels=3, classes=9).to(DEV)
    else:
        model = smp.Unet("resnet50", encoder_weights="imagenet", in_channels=3, classes=9).to(DEV)
    log("architecture:", ARCH)
    model = FullRes(model).to(DEV)
    dl = DataLoader(OEM(tr, True), batch_size=8, shuffle=True, num_workers=4, drop_last=True)
    # preflight: one step + one validation tile + one city fetch, so a bug costs a minute
    xb, yb = next(iter(dl)); o = model(xb[:2].to(DEV))
    F.cross_entropy(o.float(), yb[:2].to(DEV), ignore_index=0); dice(o.float(), yb[:2].to(DEV))
    score(model, va[:1]); fetch(28.5821, 77.0590)
    log("preflight OK")
    w = torch.tensor([0.0, 3.0, 1.3, 1.2, 1.8, 1.0, 1.3, 1.0, 1.0], device=DEV)   # bare land and roads up
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=EPOCHS * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(); t0 = time.time(); hist = []
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x)
                loss = F.cross_entropy(o.float(), y, weight=w, ignore_index=0) + dice(o.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        model.eval()
        quick = score(model, va[::6]) if ep % 5 == 0 or ep == EPOCHS else None
        hist.append({"epoch": ep, "loss": round(tot / len(dl), 4), **({"val_mIoU_sample": round(quick["mIoU"], 3)} if quick else {})})
        log(hist[-1])
        if (time.time() - t0) / 60 > TIME_CAP_MIN:
            log("time cap reached"); break
    model.eval()
    ck = {"state_dict": model.state_dict(), "classes": NAMES, "arch": ARCH, "in_channels": 3,
          "mean": MEAN.tolist(), "std": STD.tolist(), "gsd_m": 0.3, "history": hist}
    torch.save(ck, WORK / f"landcover_v2_{ARCH}.pt")
    res = score(model, va)
    res["history"] = hist
    (WORK / f"landcover_v2_{ARCH}_metrics.json").write_text(json.dumps(res, indent=2))
    log("VALIDATION", json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
    return model


# --------------------------------------------------------------- Indian cities
Z, N = 19, 4
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
CITIES = [("Chandigarh Sector 22", 30.7333, 76.7794), ("Dwarka Sector 10, Delhi", 28.5821, 77.0590),
          ("HSR Layout, Bengaluru", 12.9116, 77.6389), ("Vaishali Nagar, Jaipur", 26.9124, 75.7439),
          ("Kukatpally, Hyderabad", 17.4933, 78.3996), ("Chandni Chowk, Delhi", 28.6562, 77.2310),
          ("Dharavi, Mumbai", 19.0380, 72.8538), ("Singh Nagar, Vijayawada", 16.5256, 80.6377)]


def fetch(lat, lon):
    n = 2 ** Z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    x0, y0 = int(x) - N // 2, int(y) - N // 2
    img = np.zeros((256 * N, 256 * N, 3), np.uint8)
    for j in range(N):
        for i in range(N):
            for _ in range(3):
                try:
                    r = requests.get(URL.format(z=Z, y=y0 + j, x=x0 + i), timeout=20, headers={"User-Agent": "cadastraai-research"})
                    img[j*256:(j+1)*256, i*256:(i+1)*256] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB")); break
                except Exception:
                    time.sleep(1)
    return img


def cities(lc_model):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    out = WORK / "cities"; out.mkdir(exist_ok=True)
    summary = {}
    fig, ax = plt.subplots(len(CITIES), 2, figsize=(16, 8 * len(CITIES)))
    for r, (name, lat, lon) in enumerate(CITIES):
        rgb = fetch(lat, lon)
        lc = predict(lc_model, rgb)
        vis = (0.5 * rgb + 0.5 * COLS[lc]).astype(np.uint8)
        summary[name] = {NAMES[k]: round(float((lc == k).mean() * 100), 1) for k in range(1, 9)}
        Image.fromarray(np.hstack([rgb, np.full((1024, 8, 3), 255, np.uint8), vis])).save(out / (name.split(",")[0].replace(" ", "_") + ".jpg"), quality=88)
        ax[r, 0].imshow(rgb); ax[r, 0].set_title(name, fontsize=13)
        ax[r, 1].imshow(vis); ax[r, 1].set_title(f"land cover v2 ({ARCH})", fontsize=13)
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
        log(name, summary[name])
    fig.legend(handles=[Patch(color=COLS[i] / 255, label=NAMES[i]) for i in range(1, 9)], loc="lower center", ncol=4, fontsize=13)
    fig.tight_layout(rect=(0, 0.015, 1, 1)); fig.savefig(WORK / "cities_landcover.jpg", dpi=45)
    (WORK / "cities_landcover.json").write_text(json.dumps(summary, indent=2))


def val_panel(model):
    va = pairs("val"); random.seed(9); pick = random.sample(va, 4)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 3, figsize=(15, 20))
    for r, (ip, lp) in enumerate(pick):
        im, lb = load(ip, lp); pr = predict(model, im)
        for c, (a, t) in enumerate([(im, Path(ip).stem), ((0.45 * im + 0.55 * COLS[lb]).astype(np.uint8), "label"),
                                    ((0.45 * im + 0.55 * COLS[pr]).astype(np.uint8), "model")]):
            ax[r, c].imshow(a); ax[r, c].set_title(t, fontsize=12); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.tight_layout(); fig.savefig(WORK / "val_samples.jpg", dpi=50)


if __name__ == "__main__":
    mdl = main()
    val_panel(mdl)
    cities(mdl)
    log("done")
