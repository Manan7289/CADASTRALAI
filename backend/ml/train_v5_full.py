"""
CadastraAI v5 Full Multi-Dataset ML Training Pipeline
=====================================================
Integrates:
  1. Inria Aerial Benchmark (40 images, 5000x5000px, 0.3m GSD)
  2. AI4Boundaries Netherlands (300 orthophoto chips, 1m GSD, true cadastral parcel boundary GT via tifffile)
  3. Semantic Drone Dataset (400 UAV images, 6000x4000px, nadir drone view, roof/wall/fence/paved GT)
  4. User Datasets: Vijayawada 10cm UAV orthomosaic, ISPRS Potsdam

Outputs:
  - D:/cadastraai_data/models/model_cache/road_clf.pkl
"""
import os, sys, pickle, time, subprocess
from pathlib import Path
import numpy as np
import cv2
import tifffile
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as sk_shuffle

BACKEND_DIR = Path(__file__).resolve().parent.parent

D_MODEL_DIR = Path("D:/cadastraai_data/models/model_cache")
D_MODEL_DIR.mkdir(parents=True, exist_ok=True)
D_MODEL_PATH = D_MODEL_DIR / "road_clf.pkl"

# 13-Dimensional Multi-Scale Spatial Texture Vector
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
    L   = lab[...,0].astype(int)
    a   = lab[...,1].astype(int)
    b   = lab[...,2].astype(int)
    S   = hsv[...,1].astype(int)
    labels = np.full(img_bgr.shape[:2], 2, np.int8)
    road = (S<50) & (L>65) & (L<190) & (abs(a-128)<22) & (abs(b-128)<22)
    labels[road] = 1
    return labels

ALL_X, ALL_Y = [], []

print("=" * 65)
print("CADASTRAAI v5 — FULL MULTI-DATASET CADASTRE TRAINING")
print("=" * 65)

# ── 1. Inria Aerial Benchmark (Building GT + Multi-Tile Sampling) ────────────
INRIA_IMGS = BACKEND_DIR.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "images"
INRIA_GTS  = BACKEND_DIR.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "gt"
inria_paths = sorted(INRIA_IMGS.glob("*.tif"))
print(f"\n[1/4] Processing Inria Benchmark: {len(inria_paths)} full-res images ...")

for ip in inria_paths:
    gp = INRIA_GTS / ip.name
    if not gp.exists(): continue
    img = cv2.imread(str(ip))
    gt  = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None: continue
    H, W = gt.shape
    feats  = pixel_features(img)
    labels = pixel_labels_heuristic(img)
    labels[gt > 128] = 0  # GT building plinths
    
    for _ in range(5):
        ry = np.random.randint(0, max(1, H - 500))
        rx = np.random.randint(0, max(1, W - 500))
        F  = feats[ry:ry+500, rx:rx+500].reshape(-1, 13)
        L  = labels[ry:ry+500, rx:rx+500].ravel()
        counts = np.bincount(L.astype(np.uint8), minlength=3)
        n = min(2500, *counts)
        if n < 5: continue
        idx = np.concatenate([np.random.choice(np.where(L==c)[0], n, replace=False) for c in range(3)])
        ALL_X.append(F[idx]); ALL_Y.append(L[idx])

print(f"  Inria pixels collected: {sum(len(x) for x in ALL_X):,}")

# ── 2. AI4Boundaries Netherlands (Cadastral Boundary Ground Truth) ───────────
AI4B_IMG  = Path("D:/cadastraai_data/new_datasets/ai4boundaries/images")
AI4B_MASK = Path("D:/cadastraai_data/new_datasets/ai4boundaries/masks")
ai4b_imgs = sorted(AI4B_IMG.glob("*_image.tif"))
print(f"\n[2/4] Processing AI4Boundaries (NL Cadastral Boundary GT): {len(ai4b_imgs)} chips ...")

ai4b_added = 0
for ip in ai4b_imgs:
    fid = ip.stem.replace("_image", "")
    mp  = AI4B_MASK / f"{fid}_mask.tif"
    if not mp.exists(): continue
    img = cv2.imread(str(ip))
    if img is None: continue
    try:
        mask_arr = tifffile.imread(str(mp))
        # Channel 1 is the parcel boundary line
        b_mask = mask_arr[..., 1] if mask_arr.ndim == 3 else mask_arr
    except Exception:
        continue
    
    H, W = img.shape[:2]
    feats = pixel_features(img)
    
    labels = np.full((H, W), 2, np.int8)  # default other/interior
    labels[b_mask > 0] = 1               # true cadastral boundary line
    
    F = feats.reshape(-1, 13)
    L = labels.ravel()
    b_idx = np.where(L == 1)[0]
    o_idx = np.where(L == 2)[0]
    if len(b_idx) < 10 or len(o_idx) < 10: continue
    n = min(1500, len(b_idx), len(o_idx))
    idx = np.concatenate([
        np.random.choice(b_idx, n, replace=False),
        np.random.choice(o_idx, n, replace=False)
    ])
    ALL_X.append(F[idx]); ALL_Y.append(L[idx])
    ai4b_added += len(idx)

print(f"  AI4Boundaries pixels collected: {ai4b_added:,}")

# ── 3. Semantic Drone Dataset (Graz UAV Nadir Imagery, 400 images) ───────────
SDD_BASE = Path("D:/cadastraai_data/new_datasets/semantic_drone/dataset/semantic_drone_dataset")
sdd_imgs = sorted((SDD_BASE / "original_images").glob("*.jpg"))
print(f"\n[3/4] Processing Semantic Drone Dataset (UAV Nadir Flights): {len(sdd_imgs)} drone images ...")

sdd_added = 0
bldg_classes = {9, 10, 11, 12}
road_fence_classes = {1, 4, 13, 14}

for ip in sdd_imgs:
    lp = SDD_BASE / "label_images_semantic" / (ip.stem + ".png")
    if not lp.exists(): continue
    img = cv2.imread(str(ip))
    lbl = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
    if img is None or lbl is None: continue
    
    # Resize from 6000x4000 to 1200x800 for efficient texture extraction
    img_small = cv2.resize(img, (1200, 800), interpolation=cv2.INTER_AREA)
    lbl_small = cv2.resize(lbl, (1200, 800), interpolation=cv2.INTER_NEAREST)
    
    feats = pixel_features(img_small)
    H_s, W_s = lbl_small.shape
    
    mapped_labels = np.full((H_s, W_s), 2, np.int8)  # other
    for c in bldg_classes:
        mapped_labels[lbl_small == c] = 0           # building
    for c in road_fence_classes:
        mapped_labels[lbl_small == c] = 1           # road / fence boundary
        
    F = feats.reshape(-1, 13)
    L = mapped_labels.ravel()
    
    c0 = np.where(L == 0)[0]
    c1 = np.where(L == 1)[0]
    c2 = np.where(L == 2)[0]
    
    if len(c0) < 5 or len(c1) < 5 or len(c2) < 5: continue
    n = min(1200, len(c0), len(c1), len(c2))
    idx = np.concatenate([
        np.random.choice(c0, n, replace=False),
        np.random.choice(c1, n, replace=False),
        np.random.choice(c2, n, replace=False)
    ])
    ALL_X.append(F[idx]); ALL_Y.append(L[idx])
    sdd_added += len(idx)

print(f"  Semantic Drone pixels collected: {sdd_added:,}")

# ── 4. Vijayawada 10cm Drone Orthomosaic Dataset ─────────────────────────────
VJ_PATH = Path("D:/cadastraai_data/custom_datasets/datasets/vijayawada/aoi_singhnagar_10cm.tif")
if VJ_PATH.exists():
    print(f"\n[4/4] Processing Vijayawada 10cm UAV Orthomosaic ...")
    vj_img = cv2.imread(str(VJ_PATH))
    if vj_img is not None:
        vj_small = cv2.resize(vj_img, (1500, 1500), interpolation=cv2.INTER_AREA)
        vj_feats = pixel_features(vj_small)
        vj_labels = pixel_labels_heuristic(vj_small)
        F = vj_feats.reshape(-1, 13)
        L = vj_labels.ravel()
        counts = np.bincount(L.astype(np.uint8), minlength=3)
        n = min(15000, *counts)
        if n >= 5:
            idx = np.concatenate([np.random.choice(np.where(L==c)[0], n, replace=False) for c in range(3)])
            ALL_X.append(F[idx]); ALL_Y.append(L[idx])
            print(f"  Vijayawada 10cm UAV pixels collected: {len(idx):,}")

# ── Fit Random Forest Classifier ─────────────────────────────────────────────
X = np.vstack(ALL_X)
y = np.concatenate(ALL_Y)
X, y = sk_shuffle(X, y, random_state=42)

print("\n" + "=" * 65)
print(f"TRAINING RANDOM FOREST CLASSIFIER")
print(f"  Total multi-dataset training samples: {len(X):,} pixels")
print(f"  Class distribution: Buildings={np.sum(y==0):,}, Road/Boundary={np.sum(y==1):,}, Other/Vegetation={np.sum(y==2):,}")
print("=" * 65)

t0 = time.time()
clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=22,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=1
)
clf.fit(X, y)
t_fit = time.time() - t0

print(f"\nModel fitted in {t_fit:.1f}s. Saving directly to D: drive ({D_MODEL_PATH}) ...")
with open(D_MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

size_mb = D_MODEL_PATH.stat().st_size / (1024 * 1024)
print(f"  Saved {D_MODEL_PATH} ({size_mb:.1f} MB)")

# Regenerate live survey layers
print("\nRebuilding live survey layers with the new v5 model ...")
build_script = BACKEND_DIR / "build_clear_drone_survey.py"
ret = subprocess.run([sys.executable, str(build_script)], capture_output=True, text=True)
print(ret.stdout)
if ret.stderr:
    print("Stderr:", ret.stderr)

print("=" * 65)
print("ALL TRAINING & SURVEY GENERATION COMPLETE")
print("=" * 65)
