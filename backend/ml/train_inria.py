"""Train the binary building-footprint segmenter on Inria patches.

This replaces the RandomForest pixel classifier in model.py for the building
layer. The RF had to learn "what a roof looks like" from ~50 OSM footprints in
one Igatpuri colony; this learns it from thousands of hand-digitised footprints
across five cities, so it transfers to an AOI with no labels at all -- which is
the actual deployment condition (and what makes the upload page work).

Reported metric is IoU (Jaccard), the metric both Inria and UAVid score on.

    python -m ml.train_inria --epochs 25 --batch 8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PATCH_DIR = DATA_DIR / "datasets" / "inria_patches"
OUT_DIR = DATA_DIR / "models"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InriaPatches(Dataset):
    def __init__(self, split: str, augment: bool):
        self.img_dir = PATCH_DIR / split / "images"
        self.mask_dir = PATCH_DIR / split / "masks"
        self.stems = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.augment = augment
        if not self.stems:
            raise RuntimeError(f"No patches in {self.img_dir}. Run ml.prepare_inria first.")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = np.array(Image.open(self.img_dir / f"{stem}.jpg").convert("RGB"), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(self.mask_dir / f"{stem}.png")) > 127).astype(np.float32)

        if self.augment:
            rng = np.random
            k = rng.randint(4)
            if k:
                img, mask = np.rot90(img, k, (0, 1)), np.rot90(mask, k, (0, 1))
            if rng.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if rng.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]
            # COLOUR INVARIANCE. Measured on the Igatpuri AOI, a model trained with
            # only brightness/contrast jitter reached 28% recall and the buildings
            # it found were +12 redder than the ones it missed: it had learned
            # "bright terracotta roof" from Inria's Vienna/Tyrol tiles rather than
            # "building". Igatpuri's roofs are concrete, blue metal and dark tin, so
            # that shortcut simply does not fire there.
            #
            # These augmentations deliberately destroy the roof-colour cue -- channel
            # permutation and grayscale most of all -- so the only thing left to learn
            # is shape and texture. Expect a slightly lower Inria val IoU and much
            # better transfer; transfer is the whole point here.
            if rng.rand() < 0.5:
                img = img[:, :, rng.permutation(3)]
            if rng.rand() < 0.25:
                img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
            # float32 throughout: a float64 array from rng.uniform(size=...) would
            # silently promote the image and then fail under autocast.
            gain = rng.uniform(0.7, 1.35, size=(1, 1, 3)).astype(np.float32)
            bias = rng.uniform(-0.12, 0.12, size=(1, 1, 3)).astype(np.float32)
            img = img * gain + bias
            mean = img.mean()
            img = (img - mean) * np.float32(rng.uniform(0.7, 1.4)) + mean
            img = np.clip(img, 0.0, 1.0).astype(np.float32)

            # The AOI's best available imagery is 0.562 m/px (Esri has no zoom-19
            # coverage there), so inference upsamples it ~1.9x to reach the 0.3 m/px
            # this model trains at. That input is scale-correct but soft. Randomly
            # degrading training patches the same way -- downscale, then back up --
            # is what makes the model survive that, and it is the single augmentation
            # that matters most for this transfer.
            if rng.rand() < 0.5:
                f = rng.uniform(1.4, 2.2)
                small = max(8, int(round(img.shape[0] / f)))
                img = np.asarray(Image.fromarray((img * 255).astype(np.uint8))
                                 .resize((small, small), Image.BILINEAR)
                                 .resize(img.shape[:2][::-1], Image.BILINEAR),
                                 dtype=np.float32) / 255.0

        img = ((img - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img, mask


def build_model(encoder: str = "resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1)


def dice_bce(logits, target, eps=1.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - (num / den).mean())


@torch.no_grad()
def evaluate(model, loader, device, thresh=0.5):
    model.eval()
    inter = union = 0.0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = (torch.sigmoid(model(img)) > thresh).float()
        inter += (pred * mask).sum().item()
        union += ((pred + mask) > 0).float().sum().item()
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_ds, val_ds = InriaPatches("train", True), InriaPatches("val", False)
    print(f"{len(train_ds)} train patches, {len(val_ds)} val patches")
    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl)

    model = build_model(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_iou, history, t0 = 0.0, [], time.time()

    def write_metrics(done):
        """Rewritten every epoch, not just at the end: a 30-epoch run is an hour
        of the dashboard having nothing to show, and a killed run would leave
        nothing at all."""
        (OUT_DIR / "inria_metrics.json").write_text(json.dumps({
            "dataset": "Inria Aerial Image Labeling",
            "task": "binary building footprint segmentation",
            "encoder": args.encoder,
            "best_val_iou": round(best_iou, 4),
            "epochs": args.epochs,
            "epochs_done": len(history),
            "in_progress": not done,
            "train_patches": len(train_ds),
            "val_patches": len(val_ds),
            "train_minutes": round((time.time() - t0) / 60, 1),
            "history": history,
        }, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (img, mask) in enumerate(train_dl):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = dice_bce(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 40 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}")

        iou = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_iou": iou})
        print(f"epoch {epoch}: train_loss {running / len(train_dl):.4f}  val_IoU {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "val_iou": iou, "epoch": epoch}, OUT_DIR / "inria_unet.pt")
            print(f"  saved new best (IoU {iou:.4f})")
        write_metrics(done=False)

    write_metrics(done=True)
    print(f"\nBest val IoU {best_iou:.4f} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
