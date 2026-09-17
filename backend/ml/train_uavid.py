"""Train the 8-class UAVid semantic segmenter.

This is the layer that produces the colour-coded street-scene map: building,
road, tree, vegetation, static/moving car, human, clutter. It complements the
Inria model rather than duplicating it -- Inria is nadir 0.3 m/px ortho and only
knows "building / not building", while UAVid is oblique low-altitude drone
footage, which is what the Upload page actually receives.

Scored with mean IoU over all 8 classes, the metric UAVid's own benchmark uses.

    python -m ml.train_uavid --epochs 30 --batch 8
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
PATCH_DIR = DATA_DIR / "datasets" / "uavid_patches"
OUT_DIR = DATA_DIR / "models"

CLASSES = ["Clutter", "Building", "Road", "Static Car", "Tree", "Vegetation", "Human", "Moving Car"]
N_CLASSES = len(CLASSES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class UavidPatches(Dataset):
    def __init__(self, split: str, augment: bool):
        self.img_dir = PATCH_DIR / split / "images"
        self.mask_dir = PATCH_DIR / split / "masks"
        self.stems = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        self.augment = augment
        if not self.stems:
            raise RuntimeError(f"No patches in {self.img_dir}. Run ml.prepare_uavid first.")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        img = np.array(Image.open(self.img_dir / f"{stem}.jpg").convert("RGB"), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_dir / f"{stem}.png")).astype(np.int64)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.augment:
            rng = np.random
            # Only horizontal flip: UAVid is oblique, so the sky/ground gradient
            # is meaningful and vertical flips would teach an impossible view.
            if rng.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = img * rng.uniform(0.8, 1.25) + rng.uniform(-0.1, 0.1)
            img = np.clip(img, 0.0, 1.0)

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    inter = np.zeros(N_CLASSES, dtype=np.float64)
    union = np.zeros(N_CLASSES, dtype=np.float64)
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(img).argmax(1)
        for c in range(N_CLASSES):
            p, g = pred == c, mask == c
            inter[c] += (p & g).sum().item()
            union[c] += (p | g).sum().item()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_ds, val_ds = UavidPatches("train", True), UavidPatches("val", False)
    print(f"{len(train_ds)} train patches, {len(val_ds)} val patches")
    dl = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet",
                     in_channels=3, classes=N_CLASSES).to(device)
    # Human and static car are ~1% of pixels; unweighted CE would simply never
    # predict them. Inverse-sqrt frequency is a mild correction that does not
    # destabilise the dominant classes.
    freq = np.array([0.178, 0.350, 0.146, 0.034, 0.204, 0.041, 0.008, 0.038])
    weight = torch.tensor((1.0 / np.sqrt(freq)) / (1.0 / np.sqrt(freq)).mean(),
                          dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best, history, t0 = 0.0, [], time.time()
    per_class = np.full(N_CLASSES, np.nan)

    def write_metrics(done):
        (OUT_DIR / "uavid_metrics.json").write_text(json.dumps({
            "dataset": "UAVid 2020",
            "task": "8-class semantic segmentation",
            "encoder": args.encoder,
            "best_val_miou": round(best, 4),
            "final_per_class_iou": {c: (None if np.isnan(v) else round(float(v), 4))
                                    for c, v in zip(CLASSES, per_class)},
            "epochs": args.epochs,
            "epochs_done": len(history),
            "in_progress": not done,
            "train_patches": len(train_ds), "val_patches": len(val_ds),
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
                loss = crit(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 40 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)

        miou, per_class = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_miou": miou})
        print(f"epoch {epoch}: loss {running / len(train_dl):.4f}  val_mIoU {miou:.4f}")
        print("   " + "  ".join(f"{c}={v:.2f}" for c, v in zip(CLASSES, per_class)))

        if miou > best:
            best = miou
            torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                        "classes": CLASSES, "val_miou": miou, "epoch": epoch},
                       OUT_DIR / "uavid_unet.pt")
            print(f"  saved new best (mIoU {miou:.4f})")
        write_metrics(done=False)

    write_metrics(done=True)
    print(f"\nBest val mIoU {best:.4f} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
