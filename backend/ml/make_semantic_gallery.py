"""Build the UAVid-style gallery: source frame, colour-coded label map, and the
two blended -- the three-row figure UAVid presents its own results with.

Frames are taken from the validation split wherever possible, i.e. sequences the
model was never trained on. If val has not been downloaded yet it falls back to
train frames and says so in the manifest, because a gallery of frames the model
memorised is a screenshot, not a result.

    python -m ml.make_semantic_gallery --frames 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.infer_semantic import CLASSES, PALETTE, load_model, predict_logits

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "data" / "datasets" / "uavid_raw"
MODEL_DIR = BASE / "data" / "models"
OUT = BASE / "data" / "processed" / "semantic_gallery"

PANEL_W = 720
MAX_SIDE = 1536


def pick_frames(n: int):
    for split in ("val", "train"):
        d = RAW / "images" / split
        if d.exists():
            names = sorted(p.name for p in d.glob("*.png"))
            if names:
                # Spread across sequences rather than taking 5 frames of one video.
                seqs, chosen = {}, []
                for nm in names:
                    seqs.setdefault(nm.split("_")[0], []).append(nm)
                for key in sorted(seqs, key=lambda s: (len(s), s)):
                    chosen.append((split, seqs[key][len(seqs[key]) // 2]))
                    if len(chosen) == n:
                        return chosen, split
                return chosen, split
    return [], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--checkpoint", default=str(MODEL_DIR / "uavid_unet.pt"))
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    frames, split = pick_frames(args.frames)
    if not frames:
        raise SystemExit("No UAVid frames on disk yet -- run ml.prepare_uavid first.")

    # The UAVid model queues behind the Inria one on a single GPU, so the gallery
    # is built in two passes: ground truth first (available as soon as the dataset
    # lands), predictions added once a checkpoint exists. A page showing the real
    # labelled data is far more useful than one showing a list of commands.
    ckpt = Path(args.checkpoint)
    model = ck = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ckpt.exists():
        model, ck = load_model(ckpt, device)
        print(f"loaded {ckpt} (val mIoU {ck.get('val_miou', float('nan')):.4f}) on {device}")
    else:
        print(f"no checkpoint at {ckpt} yet -- building a ground-truth-only gallery")
    print(f"using {len(frames)} frames from the {split} split")

    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for split_name, name in frames:
        img = Image.open(RAW / "images" / split_name / name).convert("RGB")
        if max(img.size) > MAX_SIDE:
            s = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
        arr = np.array(img)

        labels = colour = blend = None
        if model is not None:
            labels = predict_logits(arr, model, device, args.batch).argmax(0).astype(np.uint8)
            colour = PALETTE[labels]
            blend = (0.55 * arr + 0.45 * colour).astype(np.uint8)

        gt_path = RAW / "masks" / split_name / name
        gt_colour = gt_blend = None
        if gt_path.exists():
            gt = np.array(Image.open(gt_path).resize(img.size, Image.NEAREST))
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt_colour = PALETTE[np.where(gt > 7, 0, gt).astype(np.uint8)]
            gt_blend = (0.55 * arr + 0.45 * gt_colour).astype(np.uint8)

        stem = Path(name).stem
        h = int(round(PANEL_W * arr.shape[0] / arr.shape[1]))

        def save(a, fn, jpg=False):
            im = Image.fromarray(a).resize((PANEL_W, h), Image.NEAREST if not jpg else Image.BILINEAR)
            im.save(OUT / fn, **({"quality": 88} if jpg else {}))

        save(arr, f"{stem}_source.jpg", jpg=True)
        entry = {"stem": stem, "split": split_name, "source": f"{stem}_source.jpg",
                 "labels": None, "overlay": None, "gt": None, "gt_overlay": None}
        if colour is not None:
            save(colour, f"{stem}_labels.png")
            save(blend, f"{stem}_overlay.jpg", jpg=True)
            entry["labels"] = f"{stem}_labels.png"
            entry["overlay"] = f"{stem}_overlay.jpg"
            total = labels.size
            entry["class_fraction"] = {c: round(float((labels == i).sum()) / total, 4)
                                       for i, c in enumerate(CLASSES)}
        if gt_colour is not None:
            save(gt_colour, f"{stem}_gt.png")
            save(gt_blend, f"{stem}_gt_overlay.jpg", jpg=True)
            entry["gt"] = f"{stem}_gt.png"
            entry["gt_overlay"] = f"{stem}_gt_overlay.jpg"
        items.append(entry)
        print(f"  {stem} done")

    (OUT / "gallery.json").write_text(json.dumps({
        "split": split,
        "held_out": split == "val",
        "has_predictions": model is not None,
        "checkpoint_val_miou": ck.get("val_miou") if ck else None,
        "classes": CLASSES,
        "palette": {c: PALETTE[i].tolist() for i, c in enumerate(CLASSES)},
        "panel_width": PANEL_W,
        "items": items,
    }, indent=2), encoding="utf-8")
    print(f"-> {OUT / 'gallery.json'}")


if __name__ == "__main__":
    main()
