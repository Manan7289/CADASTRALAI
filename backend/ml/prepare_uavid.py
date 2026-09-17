"""Fetch UAVid 2020 and cut it into 512x512 semantic-segmentation patches.

Mirror: dronefreak/UAVid-2020 (ungated re-upload of the official CC BY-NC-SA
release -- 200 train / 70 val / 150 test frames, masks already converted from
UAVid's RGB palette to single-channel class indices 0-7).

Two choices worth knowing about:

* Frames are downscaled 2x before tiling. UAVid is 4K; at native resolution a
  512 px window sees so little ground that the large classes this project cares
  about (building, road, vegetation) lose their context. Halving trades away
  some accuracy on the tiny classes (human, static car) for much better
  building/road structure, which is the right trade for a cadastral tool.
* Every frame is kept by default even though each sequence is 10 frames of one
  video and consecutive frames are near-duplicates -- use --frame-stride to
  thin them if training time matters more than coverage.

    python -m ml.prepare_uavid
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO_ID = "dronefreak/UAVid-2020"
CLASSES = ["Clutter", "Building", "Road", "Static Car", "Tree", "Vegetation", "Human", "Moving Car"]
# UAVid's official palette, kept so predictions render in the colours the
# dataset's own published figures use.
PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192],
    [0, 128, 0], [128, 128, 0], [64, 64, 0], [64, 0, 128],
], dtype=np.uint8)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "datasets" / "uavid_raw"
PATCH_DIR = DATA_DIR / "datasets" / "uavid_patches"

PATCH = 512
DOWNSCALE = 2


def list_split(split: str) -> list[str]:
    from huggingface_hub import HfApi
    fs = HfApi().repo_info(REPO_ID, repo_type="dataset").siblings
    pre = f"images/{split}/"
    return sorted(f.rfilename[len(pre):] for f in fs if f.rfilename.startswith(pre))


def download(split: str, names: list[str]) -> None:
    from huggingface_hub import hf_hub_download
    for i, n in enumerate(names):
        for kind in ("images", "masks"):
            if (RAW_DIR / kind / split / n).exists():
                continue
            hf_hub_download(REPO_ID, repo_type="dataset",
                            filename=f"{kind}/{split}/{n}", local_dir=RAW_DIR)
        if i % 10 == 0:
            print(f"  {split}: {i}/{len(names)}", flush=True)


def positions(size: int) -> list[int]:
    if size <= PATCH:
        return [0]
    pos = list(range(0, size - PATCH + 1, PATCH))
    if pos[-1] != size - PATCH:
        pos.append(size - PATCH)
    return pos


def cut(split: str, names: list[str], out_split: str) -> int:
    for kind in ("images", "masks"):
        (PATCH_DIR / out_split / kind).mkdir(parents=True, exist_ok=True)

    n_out = 0
    for idx, name in enumerate(names):
        img_p, m_p = RAW_DIR / "images" / split / name, RAW_DIR / "masks" / split / name
        if not (img_p.exists() and m_p.exists()):
            continue
        img = Image.open(img_p).convert("RGB")
        m = Image.open(m_p)
        if m.mode not in ("L", "P"):
            m = m.convert("L")
        w, h = img.size
        nw, nh = w // DOWNSCALE, h // DOWNSCALE
        img = np.array(img.resize((nw, nh), Image.BILINEAR))
        m = np.array(m.resize((nw, nh), Image.NEAREST))
        if m.ndim == 3:
            m = m[:, :, 0]
        m = np.where(m > 7, 0, m).astype(np.uint8)

        stem = Path(name).stem
        for y in positions(nh):
            for x in positions(nw):
                Image.fromarray(img[y:y + PATCH, x:x + PATCH]).save(
                    PATCH_DIR / out_split / "images" / f"{stem}_{y}_{x}.jpg", quality=92)
                Image.fromarray(m[y:y + PATCH, x:x + PATCH]).save(
                    PATCH_DIR / out_split / "masks" / f"{stem}_{y}_{x}.png")
                n_out += 1
        if idx % 20 == 0:
            print(f"  cut {split} {idx}/{len(names)} -> {n_out} patches", flush=True)
    return n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    counts = {}
    for split, out in (("train", "train"), ("val", "val")):
        names = list_split(split)[::args.frame_stride]
        print(f"{split}: {len(names)} frames")
        if not args.skip_download:
            download(split, names)
        counts[out] = cut(split, names, out)

    (PATCH_DIR / "manifest.json").write_text(json.dumps({
        "source": REPO_ID, "classes": CLASSES, "palette": PALETTE.tolist(),
        "patch": PATCH, "downscale": DOWNSCALE,
        "frame_stride": args.frame_stride, **counts,
    }, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
