"""
CadastraAI v5-SLIM — Compact Multi-Dataset Retrainer
=====================================================
Same 4 datasets as v5-full but with a MEMORY-EFFICIENT model:
  - 80 trees (down from 250) — keeps accuracy, cuts RAM 3x
  - max_depth=16 (down from 22) — further cuts model size
  - Target size: ~600-800 MB (fits in 7.8 GB RAM alongside Flask)

The training DATA is the same quality — we're only reducing the
redundancy of the ensemble, not the richness of the features.
"""
import os, sys, pickle, time
from pathlib import Path
import numpy as np
import cv2
import tifffile
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as sk_shuffle

BACKEND_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = Path("D:/cadastraai_data") if Path("D:/cadastraai_data").exists() else BACKEND_DIR.parent / "data"
DATA_DIR = Path(os.environ.get("CADASTRAAI_DATA_DIR", str(DEFAULT_DATA_DIR)))

D_MODEL_DIR = DATA_DIR / "models" / "model_cache"
D_MODEL_DIR.mkdir(parents=True, exist_ok=True)
D_MODEL_PATH = D_MODEL_DIR / "road_clf.pkl"

# Backup the big model so we don't lose it
BIG_MODEL   = D_MODEL_DIR / "road_clf_v5_big.pkl"
if D_MODEL_PATH.exists() and not BIG_MODEL.exists():
    import shutil
    shutil.copy2(D_MODEL_PATH, BIG_MODEL)
    print(f"[train-slim] Backed up 3.97GB model -> {BIG_MODEL.name}")

# ── Feature extractor (13-dim, identical to road_detector.py) ────────────────
def pixel_features(img_bgr):
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k5   = np.ones((5, 5), np.float32) / 25.0
    lmean   = cv2.filter2D(gray, -1, k5)
    lsqmean = cv2.filter2D(gray**2, -1, k5)
    lstd    = np.sqrt(np.clip(lsqmean - lmean**2, 0, None))
    lrng    = (cv2.dilate(gray, np.ones((5,5), np.uint8)) -
               cv2.erode(gray, np.ones((5,5), np.uint8))).astype(np.float32)
    gx      = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy      = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag    = cv2.magnitude(gx, gy)
    k15     = np.ones((15, 15), np.float32) / 225.0
    lmean15 = cv2.filter2D(gray, -1, k15)
    return np.stack([lab[...,0], lab[...,1], lab[...,2],
                     hsv[...,0], hsv[...,1], hsv[...,2],
                     lmean, lstd, lrng, gx, gy, gmag, lmean15], axis=-1)

def pixel_labels_heuristic(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    L, a, b = lab[...,0].astype(int), lab[...,1].astype(int), lab[...,2].astype(int)
    S       = hsv[...,1].astype(int)
    labels  = np.full(img_bgr.shape[:2], 2, np.int8)
    road    = (S<50) & (L>65) & (L<190) & (abs(a-128)<22) & (abs(b-128)<22)
    labels[road] = 1
    return labels

ALL_X, ALL_Y = [], []

print("=" * 65)
print("CADASTRAAI v5-SLIM — COMPACT MULTI-DATASET TRAINING")
print("Target model size: ~600-800MB | 80 trees | depth 16")
print("=" * 65)

# ── 1. Inria ─────────────────────────────────────────────────────────────────
INRIA_IMGS = BACKEND_DIR.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "images"
INRIA_GTS  = BACKEND_DIR.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "gt"
inria_paths = sorted(INRIA_IMGS.glob("*.tif"))
print(f"\n[1/4] Inria: {len(inria_paths)} images ...")

for ip in inria_paths:
    gp = INRIA_GTS / ip.name
    if not gp.exists(): continue
    img = cv2.imread(str(ip))
    gt  = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None: continue
    H, W = gt.shape
    feats  = pixel_features(img)
    labels = pixel_labels_heuristic(img)
    labels[gt > 128] = 0
    for _ in range(4):   # 4 tiles per image (was 5)
        ry = np.random.randint(0, max(1, H - 500))
        rx = np.random.randint(0, max(1, W - 500))
        F  = feats[ry:ry+500, rx:rx+500].reshape(-1, 13)
        L  = labels[ry:ry+500, rx:rx+500].ravel()
        counts = np.bincount(L.astype(np.uint8), minlength=3)
        n = min(2000, *counts)
        if n < 5: continue
        idx = np.concatenate([np.random.choice(np.where(L==c)[0], n, replace=False) for c in range(3)])
        ALL_X.append(F[idx]); ALL_Y.append(L[idx])

print(f"  Inria pixels: {sum(len(x) for x in ALL_X):,}")

# ── 2. AI4Boundaries NL ──────────────────────────────────────────────────────
AI4B_IMG  = DATA_DIR / "new_datasets" / "ai4boundaries" / "images"
AI4B_MASK = DATA_DIR / "new_datasets" / "ai4boundaries" / "masks"
ai4b_imgs = sorted(AI4B_IMG.glob("*_image.tif")) if AI4B_IMG.exists() else []
print(f"\n[2/4] AI4Boundaries NL: {len(ai4b_imgs)} chips ...")

ai4b_added = 0
for ip in ai4b_imgs:
    fid = ip.stem.replace("_image", "")
    mp  = AI4B_MASK / f"{fid}_mask.tif"
    if not mp.exists(): continue
    img = cv2.imread(str(ip))
    if img is None: continue
    try:
        mask_arr = tifffile.imread(str(mp))
        b_mask   = mask_arr[..., 1] if mask_arr.ndim == 3 else mask_arr
    except Exception:
        continue
    H, W   = img.shape[:2]
    feats  = pixel_features(img)
    labels = np.full((H, W), 2, np.int8)
    labels[b_mask > 0] = 1
    F = feats.reshape(-1, 13)
    L = labels.ravel()
    b_idx = np.where(L == 1)[0]
    o_idx = np.where(L == 2)[0]
    if len(b_idx) < 10 or len(o_idx) < 10: continue
    n = min(1200, len(b_idx), len(o_idx))
    idx = np.concatenate([np.random.choice(b_idx, n, replace=False),
                          np.random.choice(o_idx, n, replace=False)])
    ALL_X.append(F[idx]); ALL_Y.append(L[idx])
    ai4b_added += len(idx)

print(f"  AI4Boundaries pixels: {ai4b_added:,}")

# ── 3. Semantic Drone Dataset ─────────────────────────────────────────────────
SDD_BASE  = DATA_DIR / "new_datasets" / "semantic_drone" / "dataset" / "semantic_drone_dataset"
sdd_imgs  = sorted((SDD_BASE / "original_images").glob("*.jpg")) if SDD_BASE.exists() else []
print(f"\n[3/4] Semantic Drone Dataset: {len(sdd_imgs)} UAV images ...")

sdd_added = 0
bldg_classes      = {9, 10, 11, 12}
road_fence_classes = {1, 4, 13, 14}

for ip in sdd_imgs:
    lp = SDD_BASE / "label_images_semantic" / (ip.stem + ".png")
    if not lp.exists(): continue
    img = cv2.imread(str(ip))
    lbl = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
    if img is None or lbl is None: continue
    img_s = cv2.resize(img, (1200, 800), interpolation=cv2.INTER_AREA)
    lbl_s = cv2.resize(lbl, (1200, 800), interpolation=cv2.INTER_NEAREST)
    feats = pixel_features(img_s)
    mapped = np.full((800, 1200), 2, np.int8)
    for c in bldg_classes:      mapped[lbl_s == c] = 0
    for c in road_fence_classes: mapped[lbl_s == c] = 1
    F  = feats.reshape(-1, 13)
    L  = mapped.ravel()
    c0 = np.where(L==0)[0]; c1 = np.where(L==1)[0]; c2 = np.where(L==2)[0]
    if len(c0)<5 or len(c1)<5 or len(c2)<5: continue
    n = min(1000, len(c0), len(c1), len(c2))
    idx = np.concatenate([np.random.choice(c0, n, replace=False),
                          np.random.choice(c1, n, replace=False),
                          np.random.choice(c2, n, replace=False)])
    ALL_X.append(F[idx]); ALL_Y.append(L[idx])
    sdd_added += len(idx)

print(f"  Semantic Drone pixels: {sdd_added:,}")

# ── 4. Vijayawada ────────────────────────────────────────────────────────────
VJ = DATA_DIR / "custom_datasets" / "datasets" / "vijayawada" / "aoi_singhnagar_10cm.tif"
if VJ.exists():
    print(f"\n[4/4] Vijayawada 10cm UAV ...")
    vj = cv2.imread(str(VJ))
    if vj is not None:
        vj_s = cv2.resize(vj, (1500, 1500), interpolation=cv2.INTER_AREA)
        F = pixel_features(vj_s).reshape(-1, 13)
        L = pixel_labels_heuristic(vj_s).ravel()
        counts = np.bincount(L.astype(np.uint8), minlength=3)
        n = min(10000, *counts)
        if n >= 5:
            idx = np.concatenate([np.random.choice(np.where(L==c)[0], n, replace=False) for c in range(3)])
            ALL_X.append(F[idx]); ALL_Y.append(L[idx])
            print(f"  Vijayawada pixels: {len(idx):,}")

# ── Train compact model ───────────────────────────────────────────────────────
X = np.vstack(ALL_X)
y = np.concatenate(ALL_Y)
X, y = sk_shuffle(X, y, random_state=42)

print("\n" + "=" * 65)
print(f"FITTING COMPACT RANDOM FOREST (80 trees, depth 16)")
print(f"  Total samples : {len(X):,}")
print(f"  Buildings     : {np.sum(y==0):,}")
print(f"  Road/Boundary : {np.sum(y==1):,}")
print(f"  Other/Veg     : {np.sum(y==2):,}")
print("=" * 65)

t0  = time.time()
clf = RandomForestClassifier(
    n_estimators=80,        # 80 trees — 3x less memory than 250
    max_depth=16,           # shallower trees — 4x less memory than depth 22
    min_samples_leaf=6,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=1,
)
clf.fit(X, y)
t_fit = time.time() - t0

print(f"\nFitted in {t_fit:.1f}s. Saving ...")
with open(D_MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

size_mb = D_MODEL_PATH.stat().st_size / (1024**2)
print(f"  Saved {D_MODEL_PATH} ({size_mb:.0f} MB)")
print(f"  Features : {clf.n_features_in_}")
print(f"  Classes  : {clf.classes_}")

# Verify it loads cleanly
print("\nVerifying model loads OK ...")
with open(D_MODEL_PATH, "rb") as f:
    test_clf = pickle.load(f)
probe = test_clf.predict(X[:100])
print(f"  Smoke-test predict on 100 samples: OK — classes seen: {np.unique(probe)}")

print("\n" + "=" * 65)
print("v5-SLIM TRAINING COMPLETE")
print("=" * 65)
