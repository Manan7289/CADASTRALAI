# Parcel-boundary model v1

U-Net (ResNet34) that predicts where parcel boundaries run (compound walls, fences, hedges,
party walls between row houses). Used by the parcel step: each plot grows from its house and
stops at these lines (backend/plot_layout.py, `_nearest_in_block`).

- Training: 396 tiles (300 m, 0.3 m/px) from 11 Dutch cities. Images: PDOK Luchtfoto RGB.
  Labels: Kadaster BRK cadastral parcels (open data). Script: training/parcel_boundary/train_boundary.py (Kaggle GPU, 1 h).
- Pixel test on Groningen (held-out city), 1 m tolerance: precision 0.42, recall 0.45, F 0.43.
- Parcel test on Groningen (training/parcel_boundary/eval_parcels_nl.py): official building
  footprints (BAG) + our land cover for roads, scored against the official parcels.
  See groningen_parcel_eval.json. With the model: 43% of private parcels matched at IoU >= 0.5
  (34% without it), mean IoU 0.41 (0.37 without).

Limits: trained on Dutch imagery. Indian compound walls are often thinner and hidden by
trees; the model's lines on Indian scenes are softer, so the parcel step still falls back to
"midway between houses" where no line is found. Fine-tuning on a few Indian tiles with drawn
boundaries is the next step.
