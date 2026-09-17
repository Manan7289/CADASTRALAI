# CadastrAI — Setup Guide for New Machine

## Quick Start

```bash
git clone https://github.com/Manan7289/CADASTRALAI.git
cd CADASTRALAI
git checkout watershed-parcel-ml-v1
pip install -r requirements.txt
```

## Step 1 — Place the Model Files

Download `cadastraai_models_v1.zip` and extract into:
```
backend/ml/model_cache/
    road_clf.pkl      (76 MB — road pixel classifier)
    bldg_clf.pkl      (0.2 MB — building type classifier)
```

Or retrain from scratch (requires the Inria dataset):
```bash
cd backend
python -c "from ml.road_detector import train; train(force=True)"
python -c "from ml.building_classifier import train; train(force=True)"
```

## Step 2 — Place the Drone Image

Put your drone image at:
```
data/processed/austin_sample.jpg
```
And the Inria GT mask at:
```
data/datasets/inria_raw/data/train/gt/austin1.tif
```

## Step 3 — Run

```bash
cd backend
python build_clear_drone_survey.py   # generates parcels
python app.py                        # starts Flask server at http://127.0.0.1:5050
```

---

## Will It Run on a Different Machine?

| Scenario | Works? | Notes |
|---|---|---|
| Same Python 3.12, same sklearn 1.7.2 | ✅ Yes | Direct copy |
| Different sklearn version | ⚠️ Maybe | Retrain models with `force=True` |
| Linux / Mac | ✅ Yes | Path separators handled by `pathlib` |
| Python 3.10 or 3.11 | ✅ Yes | Just install exact packages from requirements.txt |
| No Inria dataset | ✅ Yes | Only needed to retrain — not to run inference |

## ML Pipeline Summary

```
Image (drone orthomosaic)
    ↓
road_detector.py  (Random Forest, 9 pixel features)
    → road_mask [any shape road corridors]
    → freeway_poly [largest wide corridor]
    → street_segs [Hough line segments]
    ↓
building_classifier.py  (KMeans k=3 on shape features)
    → residential / commercial / shed per building
    ↓
Watershed parcel delineation (cv2.watershed)
    → each building seeds a flood-fill
    → roads act as barriers
    → result: actual organic parcel shapes (not rectangles)
    ↓
parcels.geojson  (ULPIN + property cards + DXF)
```
