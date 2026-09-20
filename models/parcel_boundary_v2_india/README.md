# Parcel-boundary model v2 (India fine-tune)

v1 (Dutch cadastre, see ../parcel_boundary_v1) fine-tuned on hand-labelled Indian plots:
167 plot polygons over 10 crops (Singh Nagar drone ORI, Jaipur and HSR Layout satellite, 0.3 m),
labels in training/parcel_boundary/india_labels. 7 crops train, 3 held out (one per area).
Mixed 50/50 with Dutch replay tiles; 2000 steps on Kaggle (training/parcel_boundary/finetune_india.py).

| Test | v1 (Dutch only) | v2 (+ India) |
|---|---|---|
| Held-out Indian crops, boundary F @ 1 m | 0.49 | **0.61** |
| — Singh Nagar (drone) | 0.59 | 0.70 |
| — Jaipur (satellite) | 0.58 | 0.71 |
| — HSR Layout (leafy, satellite) | 0.30 | 0.41 |
| Groningen parcels matched (IoU >= 0.5) | 43% | **53%** |

Caveat: the Indian test set is small (3 crops, ~50 plots) and labelled by the same hand as the
training crops. Approved parcels from real surveys (Export > Teach the parcel model) add labels
for the next fine-tune.
