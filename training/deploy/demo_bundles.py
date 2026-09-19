"""Run the shipped models on demo areas and write one "bundle" per area for the web app.

Heavy inference stays on Kaggle; the app imports a bundle (backend/import_bundle.py)
and builds parcels, tiles and layers from it locally.

Models:
    roofs       D+ stack: our U-Net (2 folds) + teammate Inria and UAVid maps -> 8-channel Mask R-CNN
    land cover  SegFormer-B2 on OpenEarthMap (land cover v2)

Imagery: Esri World Imagery, zoom 19 (~0.3 m), warped to the area's UTM zone at 0.3 m.
Bundle (npz): rgb uint8 HxWx3, valid bool, roofs int32 (one id per house), lc_probs uint8
9xHxW (probability x 255), info json (crs, affine transform, name, sources).
"""
import glob, io, json, math, os, sys, time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/kaggle/working")
import bakeoff_friend as bf  # noqa: E402  (loads roofs v1 helpers on import)
cf, log, DEV = bf.cf, bf.log, bf.DEV
import segmentation_models_pytorch as smp  # noqa: E402
import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_bounds, from_origin  # noqa: E402
from rasterio.warp import reproject, Resampling  # noqa: E402

WORK = Path("/kaggle/working")
AREAS = [("HSR Layout, Bengaluru", 12.9116, 77.6389), ("Vaishali Nagar, Jaipur", 26.9124, 75.7439),
         ("Dwarka Sector 10, Delhi", 28.5821, 77.0590), ("Chandigarh Sector 22", 30.7333, 76.7794)]
Z, N, GSD = 19, 6, 0.3
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
HALF = 20037508.342789244
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32); IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def fetch_georef(lat, lon):
    """N x N zoom-19 tiles around the point, with their exact Web Mercator bounds."""
    n = 2 ** Z
    fx = (lon + 180) / 360 * n
    fy = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    x0, y0 = int(fx) - N // 2, int(fy) - N // 2
    img = np.zeros((256 * N, 256 * N, 3), np.uint8)
    for j in range(N):
        for i in range(N):
            for _ in range(3):
                try:
                    r = requests.get(URL.format(z=Z, y=y0 + j, x=x0 + i), timeout=20, headers={"User-Agent": "cadastraai-research"})
                    img[j*256:(j+1)*256, i*256:(i+1)*256] = np.array(Image.open(io.BytesIO(r.content)).convert("RGB")); break
                except Exception:
                    time.sleep(1)
    size = 2 * HALF / n
    left, top = -HALF + x0 * size, HALF - y0 * size
    return img, from_bounds(left, top - N * size, left + N * size, top, 256 * N, 256 * N)


def to_utm(img, t3857, lat, lon):
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    src = CRS.from_epsg(3857); dst = CRS.from_epsg(epsg)
    h, w = img.shape[:2]
    xs = [t3857.c, t3857.c + w * t3857.a]; ys = [t3857.f, t3857.f + h * t3857.e]
    from pyproj import Transformer
    tf = Transformer.from_crs(3857, epsg, always_xy=True)
    ux, uy = tf.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    # largest square, multiple of 32 px, inside the warped footprint
    side = int((min(max(ux) - min(ux), max(uy) - min(uy)) * 0.92) / GSD) // 32 * 32
    cx, cy = (min(ux) + max(ux)) / 2, (min(uy) + max(uy)) / 2
    t = from_origin(cx - side * GSD / 2, cy + side * GSD / 2, GSD, GSD)
    out = np.zeros((4, side, side), np.uint8)
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    for k in range(4):
        reproject(np.ascontiguousarray(rgba[..., k]), out[k], src_transform=t3857, src_crs=src, dst_transform=t,
                  dst_crs=dst, resampling=Resampling.bilinear if k < 3 else Resampling.nearest)
    return out[:3].transpose(1, 2, 0).copy(), out[3] > 0, t, dst


class FullRes(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(x)
        return o if o.shape[-2:] == x.shape[-2:] else F.interpolate(o, size=x.shape[-2:], mode="bilinear", align_corners=False)


def load_landcover():
    p = glob.glob("/kaggle/input/**/landcover_v2_segformer_b2.pt", recursive=True)[0]
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = FullRes(smp.Segformer("mit_b2", encoder_weights=None, in_channels=3, classes=9))
    net.load_state_dict(ck["state_dict"])
    log("land cover:", p)
    return net.to(DEV).eval()


@torch.no_grad()
def lc_probs(net, rgb, tile=512, stride=384):
    h, w = rgb.shape[:2]
    x = torch.from_numpy(((rgb.astype(np.float32) / 255 - IMNET_MEAN) / IMNET_STD).transpose(2, 0, 1)[None]).to(DEV)
    acc = torch.zeros((9, h, w), device=DEV); cnt = torch.zeros((1, h, w), device=DEV)
    ys = list(range(0, h - tile + 1, stride)) + ([h - tile] if (h - tile) % stride else [])
    xs = list(range(0, w - tile + 1, stride)) + ([w - tile] if (w - tile) % stride else [])
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x[:, :, y:y + tile, xx:xx + tile]).float()
            o[:, 0] = -1e4                          # "unlabelled" is not a real class
            acc[:, y:y + tile, xx:xx + tile] += o.softmax(1)[0]; cnt[:, y:y + tile, xx:xx + tile] += 1
    return (acc / cnt).cpu().numpy()


def main():
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    d8w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_dplus.pt", recursive=True)[0]
    inria, uavid = bf.load_friend(fr / "unet_inria_best.pt"), bf.load_friend(fr / "uavid_unet_best.pt")
    nets = [cf.load_unet(bf.ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    d8 = cf.build_maskrcnn(8); d8.load_state_dict(torch.load(d8w, map_location="cpu")); d8.eval()
    lc = load_landcover()
    out = WORK / "bundles"; out.mkdir(exist_ok=True)
    tmp = Path("/kaggle/tmp/b"); tmp.mkdir(parents=True, exist_ok=True)
    for name, lat, lon in AREAS:
        img, t3857 = fetch_georef(lat, lon)
        rgb, valid, t, crs = to_utm(img, t3857, lat, lon)
        rgb[~valid] = 0
        ps, pe, pi, _ = cf.unet_outputs(nets, rgb)
        m5 = np.dstack([cf.maps_u8(ps, pe, pi), (bf.friend_prob(inria, rgb) * 255).astype(np.uint8),
                        (bf.friend_prob(uavid, rgb, cls=1) * 255).astype(np.uint8)])
        stem = "".join(c for c in name if c.isalnum())[:30]
        np.save(tmp / f"{stem}.npy", m5)
        np.savez(tmp / f"{stem}.npz", rgb=rgb, inst=np.zeros(rgb.shape[:2], np.int16))
        roofs, _ = cf.maskrcnn_labels(d8, tmp / f"{stem}.npz", tmp, True)
        roofs[~valid] = 0
        lcp = lc_probs(lc, rgb)
        info = {"name": name, "lat": lat, "lon": lon, "crs": crs.to_string(), "transform": list(t)[:6], "gsd_m": GSD,
                "imagery": "Esri World Imagery, zoom 19 (~0.3 m) (c) Esri, Maxar",
                "models": {"roofs": "D+ stack: U-Net (roofs v1 folds) + teammate Inria/UAVid maps -> 8-ch Mask R-CNN",
                           "land_cover": "SegFormer-B2, OpenEarthMap (land cover v2)"}}
        np.savez_compressed(out / f"{stem}.npz", rgb=rgb, valid=valid, roofs=roofs.astype(np.int32),
                            lc_probs=np.clip(lcp * 255, 0, 255).astype(np.uint8), info=json.dumps(info))
        log(name, "| grid", rgb.shape, "| roofs", len(np.unique(roofs)) - 1,
            "| land cover", {k: round(float((lcp.argmax(0) == k)[valid].mean() * 100), 1) for k in range(1, 9)})
    log("done")


if __name__ == "__main__":
    main()
