"""Put Gandhinagar's overlapping tiles back on one map and split it by geography,
so training and test tiles never show the same houses.

1. Every pair of tiles is checked for overlap (keypoints + a same-scale rigid fit +
   pixel agreement), in parallel.
2. Tiles are placed on a common map, one connected group at a time.
3. In the largest group, the eastern ~20% becomes the test area; tiles fully west of
   it (minus a buffer) are training; tiles crossing the line are left out.
   Tiles from other groups are training.
4. Leak check: every training tile is compared with every test tile again, including
   flipped and rotated versions (Roboflow augmentations), and dropped if it matches.
5. Tiles are converted to the npz format train.py reads, plus a map picture.

    python stitch_split_gandhinagar.py <gandhinagar_root> <out_dir>
"""
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_gandhinagar import load_pairs, to_npz  # noqa: E402

S = 512          # working scale for matching (tiles are 1024)
TEST_SHARE = 0.20
BUFFER = 60      # px at working scale between the training and test areas
_F = None
_TF = None


def feats(path, flip=None):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)
    if flip is not None:
        g = np.ascontiguousarray(flip(g))
    orb = cv2.ORB_create(nfeatures=600)
    k, d = orb.detectAndCompute(g, None)
    return [kp.pt for kp in k], d, g


def rigid(fa, fb, min_inliers=25):
    """2x3 same-scale transform mapping tile a onto tile b, or None."""
    (pa, da, ga), (pb, db, gb) = fa, fb
    if da is None or db is None or len(pa) < 10 or len(pb) < 10:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([pa[g.queryIdx] for g in good]); dst = np.float32([pb[g.trainIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    if not 0.85 < float(np.hypot(M[0, 0], M[1, 0])) < 1.18:
        return None
    w = cv2.warpAffine(ga, M, (S, S)); v = cv2.warpAffine(np.ones_like(ga), M, (S, S), flags=cv2.INTER_NEAREST) > 0
    if v.mean() < 0.15:
        return None
    a, b = w[v].astype(np.float32), gb[v].astype(np.float32)
    ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-6))
    return M if ncc > 0.8 else None


def _pair(ij):
    i, j = ij
    return i, j, rigid(_F[i], _F[j])


def _init(F, TF=None):
    global _F, _TF
    _F, _TF = F, TF


def _leak(i):
    return i, any(rigid(_F[i], tv) is not None for tf in _TF for tv in tf)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    pairs = load_pairs(root / "train") + load_pairs(root / "test")
    n = len(pairs)
    print("tiles:", n, flush=True)
    F = [feats(p[0]) for p in pairs]
    with Pool(4, initializer=_init, initargs=(F,)) as pool:
        edges = [(i, j, M) for i, j, M in pool.imap_unordered(_pair, combinations(range(n), 2), chunksize=400) if M is not None]
    print("overlapping pairs:", len(edges), flush=True)

    # place tiles: global 3x3 transform per tile, group by group (BFS)
    adj = {i: [] for i in range(n)}
    for i, j, M in edges:
        A = np.vstack([M, [0, 0, 1]])
        adj[i].append((j, A)); adj[j].append((i, np.linalg.inv(A)))
    T, group = {}, {}
    for s in range(n):
        if s in T:
            continue
        T[s] = np.eye(3); group[s] = s; q = [s]
        while q:
            a = q.pop()
            for b, A in adj[a]:
                if b not in T:        # A maps a's pixels into b's frame; b -> map = T[a] @ inv(A)
                    T[b] = T[a] @ np.linalg.inv(A); group[b] = s; q.append(b)
    corners = np.array([[0, 0, 1], [S, 0, 1], [0, S, 1], [S, S, 1]], float).T
    ext = {i: (T[i] @ corners)[:2] for i in range(n)}
    sizes = {}
    for i, g in group.items():
        sizes[g] = sizes.get(g, 0) + 1
    main_g = max(sizes, key=sizes.get)
    members = [i for i in range(n) if group[i] == main_g]
    print("groups:", len(sizes), "| largest group:", len(members), "tiles", flush=True)

    cx = np.array([ext[i][0].mean() for i in members])
    line = np.quantile(cx, 1 - TEST_SHARE)
    test, train, dropped = [], [], []
    for i in members:
        xmin, xmax = ext[i][0].min(), ext[i][0].max()
        (test if xmin >= line else train if xmax <= line - BUFFER else dropped).append(i)
    train += [i for i in range(n) if group[i] != main_g]

    # leak check against flipped / rotated copies
    flips = [None, np.fliplr, np.flipud, lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
             lambda g: np.fliplr(np.rot90(g, 1)), lambda g: np.flipud(np.rot90(g, 1))]
    test_f = [[feats(pairs[t][0], f) for f in flips] for t in test]
    with Pool(4, initializer=_init, initargs=(F, test_f)) as pool:
        leaked = [i for i, hit in pool.imap_unordered(_leak, train, chunksize=4) if hit]
    train = [i for i in train if i not in set(leaked)]
    print(f"split: train {len(train)} | test {len(test)} | left out on the line {len(dropped)} | "
          f"dropped as flipped/rotated copies of test {len(leaked)}", flush=True)

    for split, idx in (("train", train), ("test", test)):
        d = out / split; d.mkdir(parents=True, exist_ok=True)
        houses = sum(to_npz(pairs[i][0], pairs[i][1], d / f"gn_{split}_{pairs[i][0].stem}.npz") for i in idx)
        print(f"{split}: {len(idx)} tiles, {houses} houses", flush=True)
    json.dump({"train": [pairs[i][0].name for i in train], "test": [pairs[i][0].name for i in test],
               "left_out": [pairs[i][0].name for i in dropped + leaked]}, open(out / "split.json", "w"), indent=1)

    # picture of the largest group: tiles on one map, coloured by role
    allx = np.concatenate([ext[i][0] for i in members]); ally = np.concatenate([ext[i][1] for i in members])
    x0, y0 = allx.min(), ally.min()
    scale = 0.25
    W, H = int((allx.max() - x0) * scale) + 2, int((ally.max() - y0) * scale) + 2
    canvas = np.full((H, W, 3), 25, np.uint8)
    for i in members:
        img = cv2.resize(cv2.imread(str(pairs[i][0])), (S, S))
        A = T[i].copy(); A[0, 2] -= x0; A[1, 2] -= y0; A = np.diag([scale, scale, 1]) @ A
        w = cv2.warpAffine(img, A[:2], (W, H)); m = cv2.warpAffine(np.ones((S, S), np.uint8), A[:2], (W, H)) > 0
        canvas[m] = w[m]
    for idx, col in ((train, (80, 200, 80)), (test, (60, 60, 230)), (dropped, (160, 160, 160))):
        for i in idx:
            if group[i] != main_g:
                continue
            p = ((ext[i] - np.array([[x0], [y0]])) * scale).T.astype(np.int32)[[0, 1, 3, 2]]
            cv2.polylines(canvas, [p], True, col, 2)
    lx = int((line - x0) * scale)
    cv2.line(canvas, (lx, 0), (lx, H), (0, 220, 255), 3)
    cv2.imwrite(str(out / "split_map.jpg"), canvas)
    print("wrote split_map.jpg", flush=True)


if __name__ == "__main__":
    main()
