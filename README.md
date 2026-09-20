# CadastraAI

SIH 2026 · PS 26012 — AI-Based Automated Urban Parcel Mapping and Cadastral
Feature Extraction System using Drone Imagery · Team ELAVI

A surveyor uploads drone orthoimagery (optionally with a DSM/DTM). CadastraAI
segments it, extracts building footprints, roads and lanes, and parcel
polygons, works out each parcel's land use, flags buildings that encroach on a
road or a recorded boundary, validates topology, and opens the result in a
Web-GIS workbench for review, editing, comparison with existing records, field
verification and GIS export.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/app.py            # http://127.0.0.1:5050
```

The models run on Kaggle GPU (a few minutes per survey); the laptop prepares
the image and imports the result. That needs internet and the team's Kaggle
token in `~/.kaggle/kaggle.json`. Without it the app falls back to the local
checkpoints in `models/` (not in git): `landcover_v2_segformer/` and
`unet_potsdam.pt`.

For the field page on a phone (browsers only give GPS to HTTPS pages):
`CADASTRAAI_HTTPS=1 .venv/bin/python backend/app.py`, then open
`https://<laptop LAN IP>:5443/field.html` on the same Wi-Fi.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Intake | `backend/intake.py` | GeoTIFF ORI (+ DSM, + DTM) or a plain photo/video frame georeferenced from a centre point and ground width; preview; area-of-interest; background job |
| Height | `backend/dtm.py` | DSM-only surveys get a DTM from a progressive morphological ground filter; nDSM = DSM − DTM |
| Models | `backend/kaggle_jobs.py`, `training/deploy/process_bundles.py` | Runs the approved models on Kaggle GPU: roof instances, land cover, parcel-boundary map |
| Segmentation | `backend/segment.py` | Model registry and the laptop fallback; windowed read onto a 0.3 m UTM grid |
| Roof fill-in | `backend/roof_fill.py` | Buildings the roof model misses but land cover marks as building are split into houses (watershed, shape and width checks) and tagged for review |
| Corridors & roads | `backend/parcel_extract.py`, `backend/road_network.py` | Access corridors from land cover incl. paved lanes, gaps under trees bridged, centrelines classified as lane / street / main road with lengths |
| Parcels | `backend/plot_layout.py` | Each block is rotated to its grid; plots grow from the houses and stop at the parcel-boundary model's lines; land beyond reach becomes open land |
| Plot lines | `backend/regularise.py` | The shared boundary network is split into arcs and each is straightened once, so neighbours keep one straight line between them; a line that would cut a house, or that the topology checks dislike, is reverted |
| Land use | `backend/land_use.py` | Per parcel: residential, apartments, commercial, institutional, open space, vacant, water — each with the reason it was chosen |
| Checks | `backend/parcel_extract.py`, `backend/compare.py` | Building heights and storeys from the nDSM; encroachment on a road, or across a recorded boundary |
| Topology | `backend/topology.py` | Invalid, multipart, hole, overlap, duplicate, gap, sliver, too-small checks; auto-fix that gives surveyor edits priority |
| Records | `backend/reference.py`, `backend/compare.py` | Import an existing parcel layer and GNSS/ground-truth points; match / boundary differs / split-merged / not in record; corner RMSE, CE90 |
| Field | `backend/field.py`, `frontend/field.html` | Phone page: queue by distance, verdicts, notes, photos, GPS-checked on-site visits, boundary corners into the GNSS layer |
| Survey store | `backend/survey.py` | Tiles, overlays, layers, issues, edit log, undo history, split/merge, rename, soft delete |
| Report & export | `backend/report.py`, `backend/export.py` | Printable survey report; GeoPackage / Shapefile in the survey's UTM zone, GeoJSON |
| Feedback loop | `backend/training_labels.py` | Approved parcels are exported as labels the parcel-boundary model can be retrained on |

## Models (held-out results they were accepted on)

- **Roofs — v2, the shipped model.** An 8-channel Mask R-CNN over RGB + our
  U-Net maps + the teammate's Inria and UAVid maps, trained on a Gandhinagar
  sector, then fine-tuned on Indian roofs: 117 hand-labelled roofs from
  Jaipur, HSR Layout, Dwarka, Chandigarh and Singh Nagar, plus UAVPal (a
  Bhopal drone survey with 4,820 hand-drawn roofs).
  `training/roofs_india`, `training/ensemble`.

  | Held-out set | houses found | building IoU | touching pairs merged |
  |---|---|---|---|
  | Indian hand-labelled crops | 15% → **51%** | 0.52 → **0.81** | 45% → **26%** |
  | Bhopal east strip (UAVPal) | 5% → **40%** | 0.18 → **0.68** | 11% → 23% |
  | Gandhinagar fair exam | 88% → **89%** | 0.88 | 12.5% → **9%** |

  (before → after the India fine-tune.) Overlays were checked by eye
  (`training/roofs_india/compare_roofs.py`): the extra detections are mostly
  real roofs the labels draw coarser or miss, and the old model's false alarms
  on trees and scrub are gone. Known limits: some large roofs split in two;
  outlines of big irregular buildings are rough.

- **Land cover v2** — SegFormer-B2 trained on OpenEarthMap (44 countries),
  8 classes mapped onto the app's building / road / low vegetation / tree /
  other. Validation mIoU 0.67 (road 0.65, tree 0.71, grass 0.58, bare land
  0.44). `training/landcover`.

- **Parcel boundaries** — U-Net (ResNet34) trained on Dutch cadastral parcels
  (PDOK imagery + Kadaster BRK), then fine-tuned on 167 hand-labelled Indian
  plots. Boundary F on held-out Indian crops 0.49 → **0.61**. Scored against
  800 registered parcels of the Dutch national cadastre, plots grown from the
  houses alone match **34%** at IoU 0.5; stopping them at this model's lines
  matches **53%** (mean IoU 0.37 → 0.48). `training/parcel_boundary`.

- **Aerial ORI + DSM (laptop fallback)** — U-Net ResNet34 trained on ISPRS
  Potsdam: test tiles mIoU 0.723, building IoU 0.913. `training/potsdam_unet`.

Every model is also checked visually on held-out imagery before it is used;
pixel metrics alone have hidden a degenerate fine-tune before.

## The earlier demo (kept from `main`)

The pre-finale build is still here: the static pages `frontend/index.html`,
`detection.html`, `model.html`, `cadastraai-demo.html` (what Vercel serves),
the OSM + RandomForest pipeline in `backend/fetch_data.py`, `model.py`,
`parcels.py`, `rules.py`, the teammate's Inria / UAVid training scripts in
`backend/ml/`, and its own dev server `backend/legacy_demo_app.py` with
`frontend/upload-legacy.html`. Run that one with
`.venv/bin/python backend/legacy_demo_app.py`; the finale app above is
`backend/app.py`.

## Tests

```bash
cp -R data/surveys/<id> data/surveys/e2e-test    # then set "id" in its meta.json
.venv/bin/python tests/e2e_api.py e2e-test       # 46 checks against a running server
```

## Layout

```text
backend/    Flask app + pipeline modules (above)
frontend/   workbench.html (+ records.js, fieldpanel.js, history.js), upload.html, field.html
training/   Kaggle scripts: ensemble/ (roof stack), roofs_india/ (India fine-tune + UAVPal prep),
            landcover/, parcel_boundary/, deploy/ (the per-survey GPU job), potsdam_unet/
models/     checkpoints + their metrics (gitignored; the READMEs and metrics are committed)
data/       surveys/, intake/, kaggle_jobs/, surveys_trash/ (gitignored)
tests/      e2e_api.py
```

## Data credits

UAVPal (Bhopal drone survey) — Maiti, Oude Elberink & Vosselman, IEEE JSTARS
17 (2024), doi 10.1109/JSTARS.2023.3330758; data doi 10.17026/dans-z55-6gt4,
CC BY-NC-SA 4.0. OpenEarthMap (CC BY-NC-SA 4.0 / CC BY 4.0 per source), ISPRS
Potsdam, PDOK aerial imagery and Kadaster BRK parcels, OpenAerialMap drone
ORIs, OpenStreetMap contributors.
