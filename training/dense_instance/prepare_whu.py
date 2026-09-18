"""Convert the WHU building dataset (hand-digitised roofs, COCO polygons) into the
dense-instance training tiles train.py reads.

WHU tiles are 512 x 512 at 0.3 m. Its labels are complete — every building in a
tile is outlined — so unlike our weak labels, "no roof here" is known everywhere
off the buildings. Tiles are marked complete=1 so train.py can use that.

    python prepare_whu.py <whu_root_with_annotation_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from scipy import ndimage as ndi

GSD = 0.30
ERODE_PX = 2      # ~0.6 m shrink for the interior seed
EDGE_PX = 1       # ~0.3-0.6 m ring for the roof edge
DIST_CAP_M = 2.0


def convert(root: Path, split: str, out: Path) -> int:
    ann = root / "annotation" / f"{'validation' if split == 'val' else split}.json"
    coco = COCO(str(ann))
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        path = root / split / info["file_name"]
        if not path.exists():
            continue
        rgb = np.array(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        inst = np.zeros((h, w), np.int16)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for k, a in enumerate(anns, start=1):
            m = coco.annToMask(a).astype(bool)
            inst[m & (inst == 0)] = k
        building = (inst > 0).astype(np.uint8)
        interior = np.zeros_like(building)
        edge = np.zeros_like(building)
        # per-building erosion inside each building's own bounding box (143k buildings: keep it cheap)
        for v, sl in enumerate(ndi.find_objects(inst), start=1):
            if sl is None:
                continue
            sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
            m = inst[sl] == v
            interior[sl] |= ndi.binary_erosion(m, iterations=ERODE_PX).astype(np.uint8)
            edge[sl] |= (m ^ ndi.binary_erosion(m, iterations=EDGE_PX)).astype(np.uint8)
        dist = np.clip(ndi.distance_transform_edt(building > 0) / (DIST_CAP_M / GSD), 0, 1).astype(np.float32)
        f = rgb.astype(np.float32)
        veg = (((2 * f[..., 1] - f[..., 0] - f[..., 2]) / 255.0 > 0.08) & (building == 0)).astype(np.uint8)
        np.savez_compressed(out / f"whu_{split}_{Path(info['file_name']).stem}.npz",
                            rgb=rgb, inst=inst, building=building, interior=interior, edge=edge,
                            dist=dist, veg=veg, road=np.zeros_like(building),
                            complete=np.array([1]), meta=np.array([0, 0, w * GSD, GSD], np.float64))
        n += 1
    return n


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    for split in ("train", "val"):
        print(split, convert(root, split, out / split), "tiles", flush=True)
