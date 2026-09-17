"""Fetch a balanced subset of the Inria Aerial Image Labeling dataset and cut it
into 512x512 training patches.

The full train split is 13.7 GB (180 tiles of 5000x5000 at 0.3 m/px, 36 per city).
We take an equal number of tiles from all five cities rather than more tiles from
fewer cities: Inria's stated purpose is cross-region generalization, and the AOI
this project runs on (Igatpuri) resembles none of the five, so breadth of roof
material/density matters more than depth in any one city.

    python -m ml.prepare_inria --tiles-per-city 8
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

CITIES = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
REPO_ID = "blanchon/INRIA-Aerial-Image-Labeling"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "datasets" / "inria_raw"
PATCH_DIR = DATA_DIR / "datasets" / "inria_patches"

PATCH = 512
# Tiles are 5000 px; 512*9=4608 leaves a 392 px edge, so the last row/column is
# clamped flush to the border instead of being dropped.
STRIDE = 512
MIN_BUILDING_FRAC = 0.005   # below this a patch is "empty"
EMPTY_KEEP_RATE = 0.15      # keep a few empty patches so background is still learned
VAL_CITY_TILES = 2          # tiles per city held out for validation


def download(tiles_per_city: int) -> None:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        for i in range(1, tiles_per_city + 1):
            name = f"{city}{i}.tif"
            for kind in ("images", "gt"):
                if (RAW_DIR / "data" / "train" / kind / name).exists():
                    continue
                # local_dir keeps the only copy here; the default cache would
                # hold a second one and these tiles are 75 MB each.
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=f"data/train/{kind}/{name}",
                    local_dir=RAW_DIR,
                )
            print(f"  have {name}")


def tile_positions(size: int) -> list[int]:
    pos = list(range(0, size - PATCH + 1, STRIDE))
    if pos[-1] != size - PATCH:
        pos.append(size - PATCH)
    return pos


def cut(tiles_per_city: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            (PATCH_DIR / split / kind).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped_empty": 0}
    for city in CITIES:
        for i in range(1, tiles_per_city + 1):
            name = f"{city}{i}"
            base = RAW_DIR / "data" / "train"
            img_p, gt_p = base / "images" / f"{name}.tif", base / "gt" / f"{name}.tif"
            if not (img_p.exists() and gt_p.exists()):
                continue
            split = "val" if i <= VAL_CITY_TILES else "train"

            img = np.array(Image.open(img_p).convert("RGB"))
            gt = np.array(Image.open(gt_p).convert("L"))
            h, w = gt.shape

            for y in tile_positions(h):
                for x in tile_positions(w):
                    m = (gt[y:y + PATCH, x:x + PATCH] > 127).astype(np.uint8)
                    frac = float(m.mean())
                    if frac < MIN_BUILDING_FRAC and rng.random() > EMPTY_KEEP_RATE:
                        counts["skipped_empty"] += 1
                        continue
                    stem = f"{name}_{y}_{x}"
                    Image.fromarray(img[y:y + PATCH, x:x + PATCH]).save(
                        PATCH_DIR / split / "images" / f"{stem}.jpg", quality=92)
                    Image.fromarray(m * 255).save(
                        PATCH_DIR / split / "masks" / f"{stem}.png")
                    counts[split] += 1
            print(f"  cut {name} -> {split} (running: {counts['train']} train / {counts['val']} val)")

    (PATCH_DIR / "manifest.json").write_text(json.dumps({
        "patch": PATCH, "stride": STRIDE, "cities": CITIES,
        "tiles_per_city": tiles_per_city, "val_city_tiles": VAL_CITY_TILES,
        **counts,
    }, indent=2), encoding="utf-8")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-per-city", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        print(f"Downloading {args.tiles_per_city} tiles/city from {REPO_ID} ...")
        download(args.tiles_per_city)
    print("Cutting patches ...")
    print(json.dumps(cut(args.tiles_per_city), indent=2))


if __name__ == "__main__":
    main()
