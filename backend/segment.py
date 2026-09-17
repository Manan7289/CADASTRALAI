"""Run the trained segmentation model on a georeferenced survey.

Inputs: an orthoimage GeoTIFF (ORI) and, optionally, DSM + DTM GeoTIFFs (or a
ready nDSM). Everything is warped onto one grid in the AOI's UTM zone at the
model's ground resolution (10 cm), so pixel areas are true square metres and
DSM/DTM are co-registered with the image before nDSM = DSM - DTM.

The model is height-optional (trained with height dropout): without a DSM
the height channel is fed as zeros, and the output says so.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from export import utm_crs_for

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "unet_potsdam.pt"
NDSM_SCALE_M = 30.0   # Potsdam's normalised DSM jpgs map 0..255 to roughly 0..30 m above ground
_model_cache = {}


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path=MODEL_PATH):
    key = str(path)
    if key not in _model_cache:
        import segmentation_models_pytorch as smp
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = smp.Unet("resnet34", encoder_weights=None, in_channels=ckpt["in_channels"], classes=len(ckpt["classes"]))
        m.load_state_dict(ckpt["state_dict"])
        m.eval().to(device())
        _model_cache[key] = (m, ckpt)
    return _model_cache[key]


def warp_to_grid(path, dst_crs, dst_transform, width, height, bands=None, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        out = np.zeros((len(idx), height, width), dtype=np.float32)
        for k, b in enumerate(idx):
            reproject(rasterio.band(src, b), out[k], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
                      src_nodata=src.nodata, dst_nodata=np.nan if src.nodata is not None else None)
    return out


def prepare_grid(ori_path, gsd_m):
    with rasterio.open(ori_path) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        dst_crs = utm_crs_for((lon0 + lon1) / 2, (lat0 + lat1) / 2)
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height,
                                                               *src.bounds, resolution=gsd_m)
    return dst_crs, transform, width, height


def load_survey(ori_path, dsm_path=None, dtm_path=None, ndsm_path=None, gsd_m=0.10):
    crs, transform, w, h = prepare_grid(ori_path, gsd_m)
    rgb = warp_to_grid(ori_path, crs, transform, w, h, bands=[1, 2, 3], resampling=Resampling.average)
    rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    ndsm = None
    if ndsm_path:
        ndsm = warp_to_grid(ndsm_path, crs, transform, w, h, bands=[1])[0]
    elif dsm_path and dtm_path:
        ndsm = warp_to_grid(dsm_path, crs, transform, w, h, bands=[1])[0] - warp_to_grid(dtm_path, crs, transform, w, h, bands=[1])[0]
    if ndsm is not None:
        ndsm = np.clip(np.nan_to_num(ndsm), 0, NDSM_SCALE_M)
    valid = rgb.sum(-1) > 0
    return {"rgb": rgb, "ndsm": ndsm, "valid": valid, "crs": crs, "transform": transform}


@torch.no_grad()
def predict(rgb, ndsm=None, crop=512, overlap=128, model_path=MODEL_PATH):
    model, ckpt = load_model(model_path)
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    H, W = rgb.shape[:2]
    height_u8 = np.zeros((H, W), np.float32) if ndsm is None else ndsm / NDSM_SCALE_M * 255.0
    x = np.dstack([rgb.astype(np.float32), height_u8]) / 255.0
    x = (x - mean) / std
    if ndsm is None:
        x[..., 3] = 0.0
    pad_h, pad_w = max(0, crop - H), max(0, crop - W)
    x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = x.shape[:2]
    stride = crop - overlap
    starts = lambda n: sorted(set(list(range(0, n - crop + 1, stride)) + [n - crop]))
    probs = np.zeros((len(ckpt["classes"]), Hp, Wp), np.float32)
    weight = np.zeros((Hp, Wp), np.float32)
    ramp = np.minimum(np.arange(crop) + 1, np.arange(crop)[::-1] + 1).astype(np.float32)
    win = np.minimum.outer(ramp, ramp)
    win /= win.max()
    dev = device()
    for i in starts(Hp):
        tiles = [torch.from_numpy(np.ascontiguousarray(x[i:i + crop, j:j + crop].transpose(2, 0, 1))) for j in starts(Wp)]
        out = torch.softmax(model(torch.stack(tiles).to(dev)), 1).cpu().numpy()
        for k, j in enumerate(starts(Wp)):
            probs[:, i:i + crop, j:j + crop] += out[k] * win
            weight[i:i + crop, j:j + crop] += win
    probs /= np.maximum(weight, 1e-6)
    return probs[:, :H, :W], ckpt["classes"]
