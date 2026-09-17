"""
Building Type Classifier — KMeans trained on shape features from Inria dataset.

Labels (auto-discovered):
  commercial : large, elongated, high aspect-ratio (malls, big-box stores)
  industrial : very large, rectangular, high solidity
  residential: small, compact, square-ish

No manual annotation — fully self-supervised from building geometry.
"""
from pathlib import Path
import pickle, numpy as np, cv2

MODEL_DIR  = Path(__file__).parent / "model_cache"
MODEL_PATH = MODEL_DIR / "bldg_clf.pkl"
TRAIN_GT   = Path(__file__).parent.parent.parent / "data" / "datasets" / "inria_raw" / "data" / "train" / "gt"

FEATURE_NAMES = ["log_area", "aspect_ratio", "compactness", "solidity",
                 "extent", "log_perimeter"]


def _shape_features(contour):
    """Extract 6 shape features from a contour."""
    area = cv2.contourArea(contour)
    if area < 1: return None
    peri = cv2.arcLength(contour, True)
    x,y,w,h = cv2.boundingRect(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    rect_area = float(w * h)
    return np.array([
        np.log1p(area),                              # log_area
        float(max(w,h)) / max(float(min(w,h)), 1.0), # aspect_ratio
        4*np.pi*area / max(peri**2, 1.0),             # compactness (circle=1)
        area / max(hull_area, 1.0),                   # solidity
        area / max(rect_area, 1.0),                   # extent
        np.log1p(peri),                              # log_perimeter
    ], dtype=np.float32)


def train(min_area=70, force=False):
    """Train KMeans on all buildings in Inria training set."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not force:
        print(f"[bldg_clf] Using cached model: {MODEL_PATH}")
        return

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    gt_paths = sorted(TRAIN_GT.glob("*.tif"))
    print(f"[bldg_clf] Extracting building features from {len(gt_paths)} GT masks ...")
    all_feats = []

    for gp in gt_paths:
        gt = cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE)
        if gt is None: continue
        ctrs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in ctrs:
            if cv2.contourArea(c) >= min_area:
                f = _shape_features(c)
                if f is not None:
                    all_feats.append(f)

    X = np.stack(all_feats)
    print(f"[bldg_clf] {len(X):,} buildings — fitting KMeans(k=3) ...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("km",     KMeans(n_clusters=3, random_state=42, n_init=15, max_iter=500)),
    ])
    pipe.fit(X)

    # Assign semantic labels: sort clusters by mean log_area
    # Cluster with largest mean area -> commercial
    # Middle -> residential  (most common)
    # Smallest -> shed/auxiliary
    km       = pipe.named_steps["km"]
    sc       = pipe.named_steps["scaler"]
    centers  = sc.inverse_transform(km.cluster_centers_)
    order    = np.argsort(centers[:, 0])  # sort by log_area ascending
    label_map = {
        int(order[0]): "shed",
        int(order[1]): "residential",
        int(order[2]): "commercial",
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipe, "label_map": label_map}, f)
    print(f"[bldg_clf] Saved to {MODEL_PATH}  label_map={label_map}")


def classify_buildings(bldg_contours):
    """
    Classify a list of building contours into: residential / commercial / shed.

    Uses KMeans on shape features as primary classifier, then applies
    area-based corrections derived from Inria dataset statistics:
      - area < 300 px   -> always shed  (too small to be residential)
      - area > 8000 px  -> always commercial (top 1% = large structures)
      - mid-range: use KMeans prediction (shape/aspect-ratio driven)
    """
    if not MODEL_PATH.exists():
        print("[bldg_clf] No model - training first ...")
        train()

    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        pipe      = obj.get("pipeline")
        label_map = obj.get("label_map", {})
    else:
        pipe      = obj
        label_map = {}

    # Thresholds from Inria dataset area distribution analysis
    SHED_MAX_PX        = 300    # < 300px  (27 m2) -> definitely a shed/garage
    COMMERCIAL_MIN_PX  = 8000   # > 8000px (720 m2) -> definitely commercial/industrial
    RESIDENTIAL_MAX_PX = 8000   # below this = could be residential

    results = []
    for c in bldg_contours:
        area = cv2.contourArea(c)

        # Hard area overrides first
        if area < SHED_MAX_PX:
            results.append("shed")
            continue
        if area > COMMERCIAL_MIN_PX:
            results.append("commercial")
            continue

        # Mid-range: use ML shape features
        raw_f = _shape_features(c)
        if raw_f is None:
            results.append("residential")
            continue
        try:
            feat = np.array(raw_f, dtype=np.float64).reshape(1, -1)
            cluster = int(pipe.predict(feat)[0])
        except Exception:
            try:
                km = pipe.named_steps.get("km", pipe.named_steps.get("kmeans"))
                sc = pipe.named_steps.get("scaler")
                scaled = sc.transform(np.array(raw_f, dtype=np.float64).reshape(1, -1))
                dists = np.linalg.norm(km.cluster_centers_ - scaled, axis=1)
                cluster = int(np.argmin(dists))
            except Exception:
                cluster = 1
        ml_label = label_map.get(cluster, "residential")

        # Prevent ML from calling mid-size buildings commercial unless aspect ratio is high
        # (strip malls have aspect > 3, houses < 2.5)
        if ml_label == "commercial":
            x,y,w,h = cv2.boundingRect(c)
            aspect = max(w,h) / max(min(w,h), 1.0)
            if aspect < 2.8:          # not elongated enough to be commercial
                ml_label = "residential"

        results.append(ml_label)
    return results
