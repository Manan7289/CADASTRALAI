"""Run the Inria-trained U-Net over the AOI mosaic.

Produces three things:
  building_mask.png        white-on-black footprint mask (the Inria-style raster output)
  building_overlay.png     that mask blended over the imagery, for the dashboard
  extracted_buildings.geojson   vectorised polygons, same schema model.py wrote,
                                so parcels.py / rules.py / the frontend are unchanged

Inference runs at ~0.3 m/px (the resolution the model was trained at), which is
why this reads aoi_image_z19.png rather than the zoom-17 backdrop.

    python -m ml.infer_buildings
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.geometry import shape as shp_shape

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).resolve().parent.parent.parent
PROC_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "data" / "models"

PATCH = 512
OVERLAP = 256
THRESH = 0.5
MIN_AREA_M2 = 15.0
MAX_AREA_M2 = 5000.0
UNRECORDED_IOU_THRESH = 0.15
WEB_MAX_SIDE = 2048

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def hann2d(n: int) -> np.ndarray:
    """Cosine taper so overlapping windows blend without visible seams."""
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.outer(w, w)


def load_geo(prefix: str):
    geo = json.loads((PROC_DIR / f"{prefix}_geo.json").read_text(encoding="utf-8"))
    img = np.array(Image.open(PROC_DIR / f"{prefix}.png").convert("RGB"))
    return img, geo


def px_to_lonlat_fn(geo):
    w, h = geo["width"], geo["height"]
    lon0, lat0, lon1, lat1 = geo["lon_nw"], geo["lat_nw"], geo["lon_se"], geo["lat_se"]

    def f(px, py):
        return lon0 + (px / w) * (lon1 - lon0), lat0 + (py / h) * (lat1 - lat0)
    return f


def metres_per_px(geo):
    lat = (geo["lat_nw"] + geo["lat_se"]) / 2
    w_m = abs(geo["lon_se"] - geo["lon_nw"]) * 111320 * np.cos(np.radians(lat))
    return w_m / geo["width"]


@torch.no_grad()
def predict(img: np.ndarray, model, device, batch: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    prob = np.zeros((h, w), dtype=np.float32)
    wsum = np.zeros((h, w), dtype=np.float32)
    taper = hann2d(PATCH)
    step = PATCH - OVERLAP

    ys = list(range(0, max(1, h - PATCH + 1), step))
    xs = list(range(0, max(1, w - PATCH + 1), step))
    if ys[-1] != h - PATCH:
        ys.append(h - PATCH)
    if xs[-1] != w - PATCH:
        xs.append(w - PATCH)
    coords = [(y, x) for y in ys for x in xs]
    print(f"  {len(coords)} windows of {PATCH}px (stride {step})")

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        arr = np.stack([
            ((img[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            for y, x in chunk
        ]).transpose(0, 3, 1, 2)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = torch.sigmoid(model(t))
        out = out.float().cpu().numpy()[:, 0]
        for (y, x), p in zip(chunk, out):
            prob[y:y + PATCH, x:x + PATCH] += p * taper
            wsum[y:y + PATCH, x:x + PATCH] += taper
        if (i // batch) % 25 == 0:
            print(f"    {i}/{len(coords)}")

    return prob / np.maximum(wsum, 1e-6)


_CACHE = {}


def load_model(checkpoint: Path = None, device=None):
    """Cached so the Flask upload route does not reload weights on every request."""
    import segmentation_models_pytorch as smp

    checkpoint = Path(checkpoint or (MODEL_DIR / "inria_unet.pt"))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(checkpoint), str(device))
    if key not in _CACHE:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        model = smp.Unet(encoder_name=ck.get("encoder", "resnet34"), encoder_weights=None,
                         in_channels=3, classes=1).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _CACHE[key] = (model, ck, device)
    return _CACHE[key]


def predict_mask(img_arr: np.ndarray, m_per_px: float, checkpoint: Path = None,
                 target_gsd: float = 0.3, batch: int = 4) -> np.ndarray:
    """Binary building mask for an arbitrary RGB array, at the array's own size.

    Resamples to the model's training GSD for inference and back again, so
    callers can pass imagery at whatever resolution they actually have.
    """
    model, _, device = load_model(checkpoint)
    h, w = img_arr.shape[:2]
    factor = m_per_px / target_gsd if m_per_px else 1.0
    work = img_arr
    if abs(factor - 1.0) > 0.05:
        work = np.array(Image.fromarray(img_arr).resize(
            (max(1, int(round(w * factor))), max(1, int(round(h * factor)))), Image.BICUBIC))
    prob = predict(work, model, device, batch)
    if prob.shape != (h, w):
        prob = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                        dtype=np.float32) / 255.0
    return (prob >= THRESH).astype(np.uint8)


def vectorise(mask: np.ndarray, geo, m_per_px: float):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    to_lonlat = px_to_lonlat_fn(geo)
    px_area_m2 = m_per_px ** 2

    polys = []
    for c in contours:
        area_m2 = cv2.contourArea(c) * px_area_m2
        if area_m2 < MIN_AREA_M2 or area_m2 > MAX_AREA_M2:
            continue
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        ring = [to_lonlat(float(px), float(py)) for px, py in approx]
        ring.append(ring[0])
        poly = Polygon(ring)
        if poly.is_valid and poly.area > 0:
            polys.append(poly)
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="aoi_image_z18")
    ap.add_argument("--checkpoint", default=str(MODEL_DIR / "inria_unet.pt"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--target-gsd", type=float, default=0.3,
                    help="metres/pixel to resample to before inference (Inria's native GSD)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="probability cut; defaults to whatever ml.tune_threshold chose, else 0.5")
    args = ap.parse_args()

    model, ck, device = load_model(Path(args.checkpoint))
    print(f"loaded {args.checkpoint} (val IoU {ck.get('val_iou', float('nan')):.4f}) on {device}")

    img, geo = load_geo(args.prefix)
    src_m_per_px = metres_per_px(geo)

    # Scale, not sharpness, is what a CNN encoder keys on: a building must occupy
    # the pixel count it did in training. Esri tops out at 0.562 m/px over this
    # AOI, so resample up to the training GSD before inference.
    factor = src_m_per_px / args.target_gsd
    if abs(factor - 1.0) > 0.05:
        new_wh = (int(round(geo["width"] * factor)), int(round(geo["height"] * factor)))
        print(f"resampling {geo['width']}x{geo['height']} @ {src_m_per_px:.3f} m/px "
              f"-> {new_wh[0]}x{new_wh[1]} @ {args.target_gsd:.3f} m/px (x{factor:.2f})")
        img = np.array(Image.fromarray(img).resize(new_wh, Image.BICUBIC))
    geo = {**geo, "width": img.shape[1], "height": img.shape[0]}
    m_per_px = metres_per_px(geo)
    print(f"inference raster {geo['width']}x{geo['height']} at {m_per_px:.3f} m/px")

    threshold = args.threshold
    if threshold is None:
        sweep = PROC_DIR / "threshold_sweep.json"
        threshold = (json.loads(sweep.read_text(encoding="utf-8"))["chosen_threshold"]
                     if sweep.exists() else THRESH)
    print(f"threshold {threshold}")

    prob = predict(img, model, device, args.batch)
    binary = (prob >= threshold).astype(np.uint8)

    Image.fromarray(binary * 255).save(PROC_DIR / "building_mask.png")

    overlay = img.copy()
    overlay[binary == 1] = (0.45 * overlay[binary == 1] +
                            0.55 * np.array([255, 60, 60], dtype=np.float32)).astype(np.uint8)
    Image.fromarray(overlay).save(PROC_DIR / "building_overlay.png")

    # Transparent outside buildings, and capped in size, so the browser can lay it
    # straight over the Leaflet mosaic without shipping a 20 MP opaque PNG.
    rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
    rgba[binary == 1] = (255, 60, 60, 165)
    web_size = None
    web = Image.fromarray(rgba)
    if max(web.size) > WEB_MAX_SIDE:
        s = WEB_MAX_SIDE / max(web.size)
        web_size = (int(web.width * s), int(web.height * s))
        web = web.resize(web_size, Image.NEAREST)
    web.save(PROC_DIR / "building_mask_web.png")

    # Opaque white-on-black, the way Inria presents its reference masks, plus the
    # source frame at the identical size so the two can be flipped between or
    # split-compared pixel-for-pixel on the Model Output page.
    hard = Image.fromarray(binary * 255)
    src = Image.fromarray(img)
    if web_size:
        hard = hard.resize(web_size, Image.NEAREST)
        src = src.resize(web_size, Image.BILINEAR)
    hard.save(PROC_DIR / "building_mask_hard_web.png")
    src.save(PROC_DIR / "aoi_z18_web.jpg", quality=88)

    polys = vectorise(binary, geo, m_per_px)
    print(f"vectorised {len(polys)} footprints")

    osm_fc = json.loads((PROC_DIR / "buildings.geojson").read_text(encoding="utf-8"))
    osm_polys = [shp_shape(f["geometry"]) for f in osm_fc["features"]
                 if f["geometry"]["type"] == "Polygon"]
    osm_union = unary_union(osm_polys) if osm_polys else None

    features = []
    n_unrecorded = 0
    for i, poly in enumerate(polys):
        if osm_union is not None and not osm_union.is_empty:
            frac = poly.intersection(osm_union).area / poly.area if poly.area else 0.0
        else:
            frac = 0.0
        unrecorded = frac < UNRECORDED_IOU_THRESH
        n_unrecorded += unrecorded
        features.append({"type": "Feature",
                         "properties": {"id": i, "unrecorded": bool(unrecorded)},
                         "geometry": mapping(poly)})

    fc = json.dumps({"type": "FeatureCollection", "features": features})
    # extracted_buildings.geojson is what the rest of the pipeline reads; the
    # _unet copy is kept under its own name so compare_baseline.py can score it
    # against the preserved RandomForest output rather than against itself.
    (PROC_DIR / "extracted_buildings.geojson").write_text(fc, encoding="utf-8")
    (PROC_DIR / "extracted_buildings_unet.geojson").write_text(fc, encoding="utf-8")

    stats = {
        "source": "Inria-trained U-Net",
        "checkpoint_val_iou": ck.get("val_iou"),
        "threshold": threshold,
        "inference_m_per_px": round(m_per_px, 3),
        "predicted_buildings": len(polys),
        "unrecorded_candidates": int(n_unrecorded),
        "osm_buildings_in_aoi": len(osm_polys),
        "building_pixel_fraction": round(float(binary.mean()), 4),
    }
    (PROC_DIR / "unet_inference.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
