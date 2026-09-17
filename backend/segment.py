"""Run the trained segmentation model on a georeferenced survey.

Inputs: an orthoimage GeoTIFF (ORI) and, optionally, DSM + DTM GeoTIFFs (or a
ready nDSM). Everything is warped onto one grid in the AOI's UTM zone at the
model's ground resolution (10 cm), so pixel areas are true square metres and
DSM/DTM are co-registered with the image before nDSM = DSM - DTM.

The model is height-optional (trained with height dropout): without a DSM
the height channel is fed as zeros, and the output says so.
"""
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds

import dtm as dtm_mod
from export import utm_crs_for

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# Two checkpoints, chosen per survey. Numbers are the held-out results they were accepted on.
MODELS = {
    "aerial_dsm": {
        "file": "unet_potsdam.pt",
        "label": "Aerial ORI + DSM (trained on ISPRS Potsdam)",
        "summary": "U-Net ResNet34, RGB + height. Potsdam test tiles: mIoU 0.723, building IoU 0.913.",
        "limits": "Misses many plain grey concrete roofs in Indian settlements when used without a DSM.",
    },
    "indian_drone": {
        "file": "unet_vijayawada_ft_v2.pt",
        "label": "Indian drone imagery, colour only (Potsdam model fine-tuned on Vijayawada)",
        "summary": "Held-out Singh Nagar block vs open footprints: building IoU 0.34 -> 0.77, footprints found 13 -> 36 of 49; OSM road pixels called building 6.6%.",
        "limits": "Adjacent houses are often merged into one footprint; trees and low vegetation are under-detected.",
    },
}
# CADASTRAAI_MODEL lets a dev run point every survey at another checkpoint (e.g. the smoke-test weights)
_OVERRIDE = os.environ.get("CADASTRAAI_MODEL")
MODEL_PATH = Path(_OVERRIDE) if _OVERRIDE else MODELS_DIR / MODELS["aerial_dsm"]["file"]


def choose_model(key, has_height):
    """'auto' picks the height-trained model when the survey has a DSM, the
    Indian fine-tune otherwise. Returns (key, path, info)."""
    if key in (None, "", "auto"):
        key = "aerial_dsm" if has_height else "indian_drone"
    if key not in MODELS:
        raise ValueError(f"Unknown model '{key}'.")
    path = Path(_OVERRIDE) if _OVERRIDE else MODELS_DIR / MODELS[key]["file"]
    if not path.exists():
        raise ValueError(f"Model file {path.name} is missing from models/.")
    return key, path, MODELS[key]
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
    """Read only the part of the source that falls on the target grid (via a
    WarpedVRT, which also uses the file's overviews when downsampling), so a
    multi-GB orthomosaic never has to be loaded whole."""
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        with WarpedVRT(src, crs=dst_crs, transform=dst_transform, width=width, height=height,
                       resampling=resampling, src_nodata=src.nodata, nodata=src.nodata) as vrt:
            out = vrt.read(idx, out_dtype="float32", masked=True)
    return out.filled(np.nan)


def prepare_grid(ori_path, gsd_m, aoi_lonlat=None):
    """Target grid in the AOI's UTM zone at gsd_m. aoi_lonlat = (west, south,
    east, north) restricts it to an area of interest inside the survey."""
    with rasterio.open(ori_path) as src:
        if src.crs is None:
            raise ValueError("The orthoimage has no coordinate reference system -- georeference it first.")
        full = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    w, s, e, n = aoi_lonlat if aoi_lonlat else full
    w, s, e, n = max(w, full[0]), max(s, full[1]), min(e, full[2]), min(n, full[3])
    if w >= e or s >= n:
        raise ValueError("The selected area does not overlap the orthoimage.")
    dst_crs = utm_crs_for((w + e) / 2, (s + n) / 2)
    left, bottom, right, top = transform_bounds("EPSG:4326", dst_crs, w, s, e, n)
    width, height = int(np.ceil((right - left) / gsd_m)), int(np.ceil((top - bottom) / gsd_m))
    return dst_crs, from_origin(left, top, gsd_m, gsd_m), width, height


def load_survey(ori_path, dsm_path=None, dtm_path=None, ndsm_path=None, gsd_m=0.10, aoi_lonlat=None):
    crs, transform, w, h = prepare_grid(ori_path, gsd_m, aoi_lonlat)
    rgb = warp_to_grid(ori_path, crs, transform, w, h, bands=[1, 2, 3], resampling=Resampling.average)
    valid = np.isfinite(rgb).all(0) & (np.nan_to_num(rgb).sum(0) > 0)
    rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    ndsm, height_source = None, None
    if ndsm_path:
        ndsm, height_source = warp_to_grid(ndsm_path, crs, transform, w, h, bands=[1])[0], "nDSM supplied"
    elif dsm_path and dtm_path:
        ndsm = (warp_to_grid(dsm_path, crs, transform, w, h, bands=[1])[0]
                - warp_to_grid(dtm_path, crs, transform, w, h, bands=[1])[0])
        height_source = "DSM and DTM supplied"
    elif dsm_path:
        dsm = warp_to_grid(dsm_path, crs, transform, w, h, bands=[1])[0]
        ndsm, _ = dtm_mod.ndsm_from_dsm(np.where(np.isfinite(dsm), dsm, np.nanmin(dsm)), gsd_m)
        height_source = "DSM supplied, DTM derived by ground filter"
    if ndsm is not None:
        ndsm = np.clip(np.nan_to_num(ndsm), 0, NDSM_SCALE_M)
    return {"rgb": rgb, "ndsm": ndsm, "valid": valid, "crs": crs, "transform": transform,
            "height_source": height_source}


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
