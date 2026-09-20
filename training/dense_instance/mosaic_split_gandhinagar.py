"""Rebuild the single Gandhinagar photo that its 576 overlapping crops were cut
from, merge their labels, and split the photo by geography.

1. Pairwise overlaps (keypoints + same-scale rigid fit + pixel agreement).
2. Tile positions solved together by least squares (so small errors don't pile up
   along a chain), with outlier pairs dropped and re-solved once.
3. One mosaic image, and one label map by voting: a pixel is roof if at least a
   third of the crops covering it say so. A house one crop missed but others
   labelled is filled in.
4. The eastern strip becomes the test area, a buffer is left empty, the rest is
   training. Each side is cut into fresh tiles (train 1024 px with overlap, test
   512 px without overlap so no house is scored twice).

    python mosaic_split_gandhinagar.py <gandhinagar_root> <out_dir>
"""
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).parent))
from prepare_gandhinagar import load_pairs, to_npz_instances  # noqa: E402
import stitch_split_gandhinagar as st  # noqa: E402  (feats / rigid / pool helpers)

TEST_SHARE = 0.22
BUFFER = 150          # full-res px left empty between training and test areas
VOTE = 0.5      # half the overlapping crops must agree; a third bridged the gaps between houses


def solve_positions(n, edges, root):
    """p_a - p_b = t_ab for every overlapping pair; least squares, root fixed at 0."""
    rows, cols, vals, b = [], [], [], []
    r = 0
    for a, bb, t in edges:
        for d in (0, 1):
            rows += [r, r]; cols += [2 * a + d, 2 * bb + d]; vals += [1.0, -1.0]; b.append(t[d]); r += 1
    for d in (0, 1):
        rows.append(r); cols.append(2 * root + d); vals.append(1.0); b.append(0.0); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * n))
    p = lsqr(A, np.array(b), atol=1e-10, btol=1e-10, iter_lim=20000)[0].reshape(n, 2)
    return p


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(root / "train") + load_pairs(root / "test")
    n = len(pairs)
    F = [st.feats(p[0]) for p in pairs]
    with Pool(4, initializer=st._init, initargs=(F,)) as pool:
        raw = [(i, j, M) for i, j, M in pool.imap_unordered(st._pair, combinations(range(n), 2), chunksize=400) if M is not None]
    angles = [np.degrees(np.arctan2(M[1, 0], M[0, 0])) for _, _, M in raw]
    print(f"tiles {n} | overlapping pairs {len(raw)} | rotation between pairs: median {np.median(np.abs(angles)):.2f} deg", flush=True)

    # largest connected group
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, _ in raw:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    members = max(groups.values(), key=len)
    mset = set(members)
    c = np.array([st.S / 2, st.S / 2, 1.0])
    # M maps tile i's pixels onto tile j (x_j = x_i + t). The same ground has one map
    # position: x_i + p_i = x_j + p_j, so p_i - p_j = t.
    edges = [(i, j, (M @ c)[:2] - c[:2]) for i, j, M in raw if i in mset and j in mset]
    idx = {g: k for k, g in enumerate(members)}
    e2 = [(idx[i], idx[j], t) for i, j, t in edges]
    p = solve_positions(len(members), e2, 0)
    res = np.array([np.linalg.norm(p[a] - p[b] - t) for a, b, t in e2])
    keep = res < 6.0
    print(f"group of {len(members)} tiles | pair residual median {np.median(res):.2f}px 95% {np.percentile(res, 95):.2f}px "
          f"| {int((~keep).sum())} outlier pairs dropped", flush=True)
    p = solve_positions(len(members), [e for e, k in zip(e2, keep) if k], 0)
    res = np.array([np.linalg.norm(p[a] - p[b] - t) for (a, b, t), k in zip(e2, keep) if k])
    print(f"after re-solve: residual median {np.median(res):.2f}px 95% {np.percentile(res, 95):.2f}px (at 512 scale)", flush=True)

    # full-resolution mosaic (tiles are 1024 = 2 x the 512 matching scale)
    P = np.round(p * 2).astype(int)
    P -= P.min(0)
    W, H = int(P[:, 0].max()) + 1024, int(P[:, 1].max()) + 1024
    img = np.zeros((H, W, 3), np.uint8)
    cov = np.zeros((H, W), np.uint16); pos = np.zeros((H, W), np.uint16)
    for k, g in enumerate(members):
        x, y = P[k]
        rgb = np.array(Image.open(pairs[g][0]).convert("RGB"))
        m = np.array(Image.open(pairs[g][1])) > 0
        h, w = m.shape
        img[y:y + h, x:x + w] = rgb
        cov[y:y + h, x:x + w] += 1
        pos[y:y + h, x:x + w] += m
    label = (pos >= np.maximum(1, np.ceil(VOTE * cov))) & (cov > 0)
    covered = cov > 0
    # one id per house: house cores (eroded) as seeds, grown back inside the label so a
    # thin bridge between two neighbours can't fuse them into one shape
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    seeds, n_seeds = ndi.label(ndi.binary_erosion(label, iterations=3))
    inst = watershed(-ndi.distance_transform_edt(label), markers=seeds, mask=label)
    blobs = ndi.label(label)[1]
    print(f"houses: {n_seeds} after splitting cores vs {blobs} connected blobs before", flush=True)
    print(f"mosaic {W} x {H} px (~{W * 0.3 / 1000:.2f} x {H * 0.3 / 1000:.2f} km at ~0.3 m) | "
          f"each spot seen by {np.median(cov[covered]):.0f} crops on median", flush=True)

    # geographic split: eastern strip = test
    colcov = covered.mean(0)
    xs = np.arange(W)
    cum = np.cumsum(colcov) / colcov.sum()
    line = int(xs[np.searchsorted(cum, 1 - TEST_SHARE)])
    tiles = {"train": [], "test": []}
    for y in range(0, H - 1024 + 1, 512):
        for x in range(0, line - BUFFER - 1024 + 1, 512):
            if covered[y:y + 1024, x:x + 1024].mean() >= 0.95:
                tiles["train"].append((x, y, 1024))
    for y in range(0, H - 512 + 1, 512):
        for x in range(line, W - 512 + 1, 512):
            if covered[y:y + 512, x:x + 512].mean() >= 0.95:
                tiles["test"].append((x, y, 512))
    stats = {}
    for split, lst in tiles.items():
        d = out / split; d.mkdir(parents=True, exist_ok=True)
        houses = 0
        for x, y, s in lst:
            houses += to_npz_instances(img[y:y + s, x:x + s], inst[y:y + s, x:x + s], d / f"gnm_{split}_{x}_{y}.npz")
        stats[split] = {"tiles": len(lst), "tile_px": lst[0][2] if lst else None, "house_pieces": houses}
    cen = ndi.center_of_mass(label, inst, range(1, int(inst.max()) + 1))
    cx = np.array([c[1] for c in cen])
    stats["unique_houses_train_area"] = int((cx < line - BUFFER).sum())
    stats["unique_houses_test_area"] = int((cx >= line).sum())
    stats.update(mosaic_px=[W, H], split_x=line, buffer_px=BUFFER)
    json.dump(stats, open(out / "mosaic_split.json", "w"), indent=1)
    print(json.dumps(stats), flush=True)

    # preview: mosaic, labels in blue, split line, tiles outlined
    sc = 1600 / W
    pv = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    lb = cv2.resize(label.astype(np.uint8), pv.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    pv[lb] = (0.55 * pv[lb] + 0.45 * np.array([230, 110, 60])).astype(np.uint8)
    for split, col in (("train", (60, 200, 60)), ("test", (40, 40, 230))):
        for x, y, s in tiles[split]:
            cv2.rectangle(pv, (int(x * sc), int(y * sc)), (int((x + s) * sc), int((y + s) * sc)), col, 1)
    cv2.line(pv, (int(line * sc), 0), (int(line * sc), pv.shape[0]), (0, 220, 255), 3)
    cv2.line(pv, (int((line - BUFFER) * sc), 0), (int((line - BUFFER) * sc), pv.shape[0]), (0, 220, 255), 1)
    cv2.imwrite(str(out / "mosaic_preview.jpg"), cv2.cvtColor(pv, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for name, (zx, zy) in {"closeup_train.jpg": (int(0.28 * W), int(0.30 * H)), "closeup_test.jpg": (line + 60, int(0.45 * H))}.items():
        zx, zy = min(zx, W - 1200), min(zy, H - 900)
        crop = img[zy:zy + 900, zx:zx + 1200].copy(); ic = inst[zy:zy + 900, zx:zx + 1200]
        rng = np.random.default_rng(1); cols = (rng.random((int(ic.max()) + 2, 3)) * 255).astype(np.uint8); cols[0] = 0
        mk = ic > 0
        crop[mk] = (0.55 * crop[mk] + 0.45 * cols[ic[mk]]).astype(np.uint8)
        for v in np.unique(ic[mk]):
            c, _ = cv2.findContours((ic == v).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, c, -1, (255, 255, 255), 1)
        cv2.imwrite(str(out / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote mosaic_preview.jpg", flush=True)


if __name__ == "__main__":
    main()
