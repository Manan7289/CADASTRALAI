"""Convert the Gandhinagar rooftop dataset (Roboflow export, CC BY 4.0) into the
dense-instance tiles train.py reads, and make the test split honest.

- Each binary rooftop mask is split into one instance per house (houses in this
  dataset are drawn with a gap between neighbours).
- ~10-15% of houses are unlabelled, so tiles are marked partial=1: train.py then
  treats unlabelled ground as unknown instead of "not a roof".
- Roboflow exports contain shifted/augmented copies. Any training tile that
  overlaps a test tile (ORB keypoints + RANSAC homography) is dropped, so test
  scores are not inflated by near-duplicates.

    python prepare_gandhinagar.py <gandhinagar_root> <out_dir>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GSD = 0.30          # approximate; Gandhinagar tiles are at WHU-like scale
ERODE_PX = 2
EDGE_PX = 1
DIST_CAP_M = 2.0
MIN_HOUSE_PX = 40


def load_pairs(split_dir: Path):
    out = []
    for m in sorted(split_dir.glob("*_mask.png")):
        img = m.with_name(m.name.replace("_mask.png", ".jpg"))
        if img.exists():
            out.append((img, m))
    return out


def orb_features(path: Path, orb):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
    k, d = orb.detectAndCompute(g, None)
    return k, d, g


def overlaps(fa, fb, matcher, min_inliers=25):
    """True only if the two tiles show the same ground. Planned layouts repeat, so
    keypoint matches alone are fooled by look-alike neighbourhoods: the fitted
    transform must also be a plain shift/rotation at the same scale, and the pixels
    must agree once one tile is warped onto the other."""
    (ka, da, ga), (kb, db, gb) = fa, fb
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return False
    pairs = matcher.knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return False
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return False
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not 0.85 < scale < 1.18:
        return False
    warped = cv2.warpAffine(ga, M, gb.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpAffine(np.ones_like(ga), M, gb.shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0) > 0
    if valid.mean() < 0.2:
        return False
    a = warped[valid].astype(np.float32); b = gb[valid].astype(np.float32)
    ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-6))
    return ncc > 0.8


def to_npz(img_path: Path, mask_path: Path, out_path: Path) -> int:
    rgb = np.array(Image.open(img_path).convert("RGB"))
    m = np.array(Image.open(mask_path)) > 0
    return to_npz_arrays(rgb, m, out_path)


def to_npz_arrays(rgb: np.ndarray, m: np.ndarray, out_path: Path) -> int:
    inst, _ = ndi.label(m)   # 4-connectivity: diagonal contact does not merge houses
    return to_npz_instances(rgb, inst, out_path)


def to_npz_instances(rgb: np.ndarray, inst: np.ndarray, out_path: Path) -> int:
    """inst: one id per house (touching houses keep different ids)."""
    inst = inst.astype(np.int32).copy()
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is not None and (inst[sl] == v).sum() < MIN_HOUSE_PX:
            inst[inst == v] = 0
    ids = np.unique(inst); ids = ids[ids > 0]
    remap = np.zeros(int(inst.max()) + 1, np.int32); remap[ids] = np.arange(1, len(ids) + 1)
    inst = remap[inst]; n = len(ids)
    building = (inst > 0).astype(np.uint8)
    interior = np.zeros_like(building)
    edge = np.zeros_like(building)
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
        h = inst[sl] == v
        interior[sl] |= ndi.binary_erosion(h, iterations=ERODE_PX).astype(np.uint8)
        edge[sl] |= (h ^ ndi.binary_erosion(h, iterations=EDGE_PX)).astype(np.uint8)
    dist = np.clip(ndi.distance_transform_edt(building > 0) / (DIST_CAP_M / GSD), 0, 1).astype(np.float32)
    f = rgb.astype(np.float32)
    veg = (((2 * f[..., 1] - f[..., 0] - f[..., 2]) / 255.0 > 0.08) & (building == 0)).astype(np.uint8)
    np.savez_compressed(out_path, rgb=rgb, inst=inst.astype(np.int16), building=building,
                        interior=interior, edge=edge, dist=dist, veg=veg, road=np.zeros_like(building),
                        partial=np.array([1]), meta=np.array([0, 0, rgb.shape[1] * GSD, GSD], np.float64))
    return n


if __name__ == "__main__":
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    train, test = load_pairs(root / "train"), load_pairs(root / "test")
    orb = cv2.ORB_create(nfeatures=600)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    test_f = [orb_features(i, orb) for i, _ in test]
    # Roboflow keeps the original upload's number: same "sample_<n>" = same source image
    test_ids = {i.name.split("_jpg")[0] for i, _ in test}
    keep, dropped = [], []
    for img, msk in train:
        same_source = img.name.split("_jpg")[0] in test_ids
        f = orb_features(img, orb)
        dup = same_source or any(overlaps(f, t, matcher) for t in test_f)
        (dropped if dup else keep).append((img, msk))
    print(f"train tiles: {len(train)} | overlapping a test tile, dropped: {len(dropped)} | kept: {len(keep)}", flush=True)
    for split, pairs in (("train", keep), ("test", test)):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        houses = sum(to_npz(i, m, d / f"gn_{split}_{i.stem}.npz") for i, m in pairs)
        print(f"{split}: {len(pairs)} tiles, {houses} houses", flush=True)
