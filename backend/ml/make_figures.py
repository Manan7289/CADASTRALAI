"""Assemble the comparison figures the two datasets present their results with.

  --mode inria    image | predicted mask           (and | ground truth, when known)
  --mode uavid    frame | colour label map | blend

Used for the writeup and the slide deck, so panels are labelled and laid out the
same way the dataset papers lay theirs out.

    python -m ml.make_figures --mode inria --out ../data/processed/figures
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

BASE = Path(__file__).resolve().parent.parent.parent
PROC = BASE / "data" / "processed"

LABEL_H = 34
PAD = 10
BG = (255, 255, 255)


def _font(size=20):
    for n in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def strip(panels: list[tuple[str, Image.Image]], cell: int = 520) -> Image.Image:
    imgs = [im.convert("RGB").resize((cell, cell), Image.BILINEAR) for _, im in panels]
    w = len(imgs) * cell + (len(imgs) + 1) * PAD
    out = Image.new("RGB", (w, cell + LABEL_H + 2 * PAD), BG)
    d = ImageDraw.Draw(out)
    f = _font()
    for i, (im, (title, _)) in enumerate(zip(imgs, panels)):
        x = PAD + i * (cell + PAD)
        out.paste(im, (x, PAD))
        tw = d.textlength(title, font=f)
        d.text((x + (cell - tw) / 2, PAD + cell + 6), title, fill=(20, 20, 20), font=f)
    return out


def crop_centre(im: Image.Image, frac: float) -> Image.Image:
    if frac >= 1.0:
        return im
    w, h = im.size
    cw, ch = int(w * frac), int(h * frac)
    return im.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))


def inria_figure(out_dir: Path, zoom_frac: float):
    src = Image.open(PROC / "aoi_image_z18.png")
    mask = Image.open(PROC / "building_mask.png")
    over = Image.open(PROC / "building_overlay.png")
    # The mask is produced at the resampled inference size; match the source to it.
    src = src.resize(mask.size, Image.BICUBIC)
    panels = [("Igatpuri AOI (Esri, 0.56 m/px)", crop_centre(src, zoom_frac)),
              ("Predicted footprints", crop_centre(mask, zoom_frac)),
              ("Overlay", crop_centre(over, zoom_frac))]
    fig = strip(panels)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.save(out_dir / "figure_inria_style.png")
    print("wrote", out_dir / "figure_inria_style.png")


def uavid_figure(sem_dir: Path, out_dir: Path, zoom_frac: float):
    panels = [("UAV frame", Image.open(sem_dir / "semantic_source.jpg")),
              ("Semantic labels", Image.open(sem_dir / "semantic_labels.png")),
              ("Overlay", Image.open(sem_dir / "semantic_overlay.jpg"))]
    panels = [(t, crop_centre(im, zoom_frac)) for t, im in panels]
    fig = strip(panels)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.save(out_dir / "figure_uavid_style.png")
    print("wrote", out_dir / "figure_uavid_style.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inria", "uavid"], required=True)
    ap.add_argument("--out", default=str(PROC / "figures"))
    ap.add_argument("--sem-dir", default=str(PROC / "semantic"))
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="centre-crop fraction, e.g. 0.35 to show detail")
    a = ap.parse_args()
    if a.mode == "inria":
        inria_figure(Path(a.out), a.zoom)
    else:
        uavid_figure(Path(a.sem_dir), Path(a.out), a.zoom)


if __name__ == "__main__":
    main()
