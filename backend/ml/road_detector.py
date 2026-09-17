"""
Road / Freeway Detector — learns from Inria dataset. No hardcoded coords.

Trains a Random Forest pixel classifier:
  label 0 = Building  (from GT mask)
  label 1 = Road/Asphalt
  label 2 = Vegetation / Other

Saves model to ml/model_cache/road_clf.pkl
"""
from pathlib import Path
import pickle, numpy as np, cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

MODEL_DIR   = Path(__file__).parent / "model_cache"
MODEL_PATH  = MODEL_DIR / "road_clf.pkl"
TRAIN_IMGS  = Path(__file__).parent.parent.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "images"
TRAIN_GTS   = Path(__file__).parent.parent.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "gt"


def _pixel_features(img_bgr):
    """9-dim per-pixel feature: LAB + HSV + 3 texture stats."""
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k    = np.ones((5,5), np.float32) / 25.0
    lmean = cv2.filter2D(gray, -1, k)
    lstd  = np.sqrt(np.clip(cv2.filter2D(gray**2, -1, k) - lmean**2, 0, None))
    lrng  = (cv2.dilate(gray, np.ones((5,5),np.uint8)) -
              cv2.erode(gray,  np.ones((5,5),np.uint8)))
    return np.stack([lab[...,0], lab[...,1], lab[...,2],
                     hsv[...,0], hsv[...,1], hsv[...,2],
                     lmean, lstd, lrng], axis=-1)


def _pixel_labels(gt, img_bgr):
    """Self-supervised labels from GT mask + color heuristics."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    L   = lab[...,0].astype(int)
    a   = lab[...,1].astype(int)
    b   = lab[...,2].astype(int)
    S   = hsv[...,1].astype(int)
    labels = np.full(gt.shape, 2, np.int8)                    # default: other
    road = (S<50) & (L>65) & (L<190) & (abs(a-128)<22) & (abs(b-128)<22)
    labels[road]    = 1
    labels[gt>128]  = 0  # buildings override
    return labels


def train(n_tiles=5, tile_size=500, px_per_class=2500, force=False):
    """Train on all Inria images.  Skips if model already exists."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force:
        print(f"[road_detector] Using cached model: {MODEL_PATH}")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.utils import shuffle as sk_shuffle

    img_paths = sorted(TRAIN_IMGS.glob("*.tif"))
    print(f"[road_detector] Training on {len(img_paths)} images ...")
    Xs, ys = [], []

    for ip in img_paths:
        gp = TRAIN_GTS / ip.name
        if not gp.exists(): continue
        img = cv2.imread(str(ip))
        gt  = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None: continue
        H, W = gt.shape
        feats  = _pixel_features(img)
        labels = _pixel_labels(gt, img)
        for _ in range(n_tiles):
            ry = np.random.randint(0, max(1, H-tile_size))
            rx = np.random.randint(0, max(1, W-tile_size))
            F  = feats[ry:ry+tile_size, rx:rx+tile_size].reshape(-1,9)
            L  = labels[ry:ry+tile_size, rx:rx+tile_size].ravel()
            n = min(px_per_class, *(np.bincount(L.astype(np.uint8), minlength=3)))
            if n < 5: continue
            idx = np.concatenate([np.random.choice(np.where(L==c)[0], n, replace=False) for c in range(3)])
            Xs.append(F[idx]); ys.append(L[idx])

    X = np.vstack(Xs); y = np.concatenate(ys)
    X, y = sk_shuffle(X, y, random_state=42)
    print(f"[road_detector] Fitting RF on {len(X):,} pixels ...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=14,
                                  min_samples_leaf=8, n_jobs=-1,
                                  random_state=42, class_weight="balanced")
    clf.fit(X, y)
    with open(MODEL_PATH,"wb") as f: pickle.dump(clf, f)
    print(f"[road_detector] Saved to {MODEL_PATH}")


def detect_roads(img_bgr, gt_mask=None):
    """
    Returns (road_mask uint8, freeway_poly Polygon px-coords, street_segs list).
    freeway_poly = convex hull of the largest continuous road region.
    street_segs  = [(angle_deg, (pt1, pt2)), ...]
    """
    if not MODEL_PATH.exists():
        print("[road_detector] No model — training first ...")
        train()

    with open(MODEL_PATH,"rb") as f:
        clf = pickle.load(f)

    H, W = img_bgr.shape[:2]
    feats = _pixel_features(img_bgr).reshape(-1,9)
    chunk = 400_000
    pred  = np.empty(H*W, dtype=np.int8)
    for s in range(0, H*W, chunk):
        pred[s:s+chunk] = clf.predict(feats[s:s+chunk])
    pred = pred.reshape(H, W)

    road_mask = (pred == 1).astype(np.uint8) * 255
    if gt_mask is not None:
        road_mask[gt_mask > 128] = 0

    # Morphological cleanup
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  ko)

    # ── Freeway detection ───────────────────────────────────────────────────
    # Use a large erosion kernel (60x60) so only genuinely wide corridors
    # (freeway = 6+ lanes = ~60px+ wide at 0.3m/px) survive.
    # This prevents residential streets and parking lots from being called freeway.
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    big = cv2.erode(road_mask, kb, iterations=1)
    big = cv2.dilate(big,      kb, iterations=2)

    num_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(big, 8)

    freeway_poly = Polygon([(0,0),(2,0),(2,2),(0,2)])  # tiny fallback
    MAX_FREEWAY_AREA = H * W * 0.35   # cap: freeway can't be >35% of image

    if num_lbl > 1:
        # Pick largest component that is NOT the whole image
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        valid_idx = [i for i,a in enumerate(component_areas) if a < MAX_FREEWAY_AREA]
        if valid_idx:
            best = int(np.argmax(component_areas[valid_idx])) 
            best_lbl = valid_idx[best] + 1
            fw_mask = (lbl_img == best_lbl).astype(np.uint8) * 255
            ctrs, _ = cv2.findContours(fw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if ctrs:
                hull = cv2.convexHull(np.vstack(ctrs)).reshape(-1,2).tolist()
                if len(hull) >= 3:
                    p = Polygon(hull)
                    freeway_poly = p.buffer(0) if not p.is_valid else p

    # ── Street centerlines via Hough ────────────────────────────────────────
    thin = cv2.erode(road_mask, np.ones((3,3),np.uint8), iterations=2)
    lines = cv2.HoughLinesP(thin, 1, np.pi/180, threshold=45,
                              minLineLength=55, maxLineGap=20)
    segs = []
    if lines is not None:
        for ln in lines:
            pts = np.array(ln).ravel()
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            ang = float(np.degrees(np.arctan2(y2-y1, x2-x1)) % 180)
            segs.append((ang, ((x1,y1),(x2,y2))))

    print(f"[road_detector] road_px={(road_mask>0).sum():,}  "
          f"freeway_area={freeway_poly.area:.0f}px2 ({freeway_poly.area/(H*W)*100:.1f}%)  "
          f"streets={len(segs)}")
    return road_mask, freeway_poly, segs

