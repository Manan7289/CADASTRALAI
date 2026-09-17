"""Run the UAVid-trained 8-class segmenter over an image.

Writes the three panels UAVid's own figures show: the source frame, the
colour-coded label map in UAVid's palette, and the two blended together.

    python -m ml.infer_semantic --image path/to/drone.jpg --out-dir somewhere
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE / "data" / "models"

CLASSES = ["Clutter", "Building", "Road", "Static Car", "Tree", "Vegetation", "Human", "Moving Car"]
PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
    [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128],
], dtype=np.uint8)

PATCH = 512
OVERLAP = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(checkpoint: Path, device):
    import segmentation_models_pytorch as smp
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ck.get("encoder", "resnet34"), encoder_weights=None,
                     in_channels=3, classes=len(CLASSES)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.outer(w, w)


@torch.no_grad()
def predict_logits(img: np.ndarray, model, device, batch: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h, pad_w = max(0, PATCH - h), max(0, PATCH - w)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w = img.shape[:2]

    acc = np.zeros((len(CLASSES), h, w), dtype=np.float32)
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

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        arr = np.stack([
            ((img[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
            for y, x in chunk
        ]).transpose(0, 3, 1, 2)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = torch.softmax(model(t), dim=1)
        out = out.float().cpu().numpy()
        for (y, x), p in zip(chunk, out):
            acc[:, y:y + PATCH, x:x + PATCH] += p * taper
            wsum[y:y + PATCH, x:x + PATCH] += taper

    acc /= np.maximum(wsum, 1e-6)
    return acc[:, :h - pad_h or None, :w - pad_w or None]


def colourise(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels]


def run(image_path: Path, out_dir: Path, checkpoint: Path = None, batch: int = 4,
        max_side: int = 2048) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or (MODEL_DIR / "uavid_unet.pt")
    model, ck = load_model(checkpoint, device)

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    arr = np.array(img)

    prob = predict_logits(arr, model, device, batch)
    labels = prob.argmax(0).astype(np.uint8)
    colour = colourise(labels)
    blend = (0.55 * arr + 0.45 * colour).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_dir / "semantic_source.jpg", quality=92)
    Image.fromarray(colour).save(out_dir / "semantic_labels.png")
    Image.fromarray(blend).save(out_dir / "semantic_overlay.jpg", quality=92)

    total = labels.size
    stats = {
        "checkpoint_val_miou": ck.get("val_miou"),
        "size": [int(arr.shape[1]), int(arr.shape[0])],
        "class_fraction": {c: round(float((labels == i).sum()) / total, 4)
                           for i, c in enumerate(CLASSES)},
        "palette": {c: PALETTE[i].tolist() for i, c in enumerate(CLASSES)},
    }
    (out_dir / "semantic_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out-dir", default=str(BASE / "data" / "processed" / "semantic"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=2048)
    a = ap.parse_args()
    ck = Path(a.checkpoint) if a.checkpoint else None
    print(json.dumps(run(Path(a.image), Path(a.out_dir), ck, a.batch, a.max_side), indent=2))


if __name__ == "__main__":
    main()
