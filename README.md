# CadastraAI

SIH 2026 · PS 26012 — AI-Based Automated Urban Parcel Mapping and Cadastral
Feature Extraction System using Drone Imagery · Team ELAVI

A surveyor uploads drone orthoimagery (optionally with a DSM/DTM). CadastraAI
segments it, extracts building footprints, access corridors and parcel
polygons, validates their topology, and opens the result in a Web-GIS
workbench for review, editing, comparison with existing records, field
verification and GIS export.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python backend/app.py            # http://127.0.0.1:5050
```

Model checkpoints live in `models/` (not in git): `unet_potsdam.pt` and
`unet_vijayawada_ft_v2.pt`. For the field page on a phone (browsers only give
GPS to HTTPS pages): `CADASTRAAI_HTTPS=1 .venv/bin/python backend/app.py`,
then open `https://<laptop LAN IP>:5443/field.html` on the same Wi-Fi.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| Intake | `backend/intake.py` | GeoTIFF ORI (+ DSM, + DTM) or a plain photo/video frame georeferenced from a centre point and ground width; preview; area-of-interest; background job |
| Height | `backend/dtm.py` | DSM-only surveys get a DTM from a progressive morphological ground filter; nDSM = DSM − DTM |
| Segmentation | `backend/segment.py` | Windowed read onto a 10 cm UTM grid; U-Net (ResNet34) with building / road / low vegetation / tree / other classes, RGB + optional height |
| Extraction | `backend/parcel_extract.py` | Access corridors (width, narrow lanes), roof instances split on colour and height edges, parcels grown from roofs and split over open land, one gap-free coverage, regularised shapes, per-parcel confidence with reasons |
| Topology | `backend/topology.py` | Invalid, multipart, hole, overlap, duplicate, gap, sliver, too-small checks; auto-fix that gives surveyor edits priority |
| Records | `backend/reference.py`, `backend/compare.py` | Import an existing parcel layer and GNSS/ground-truth points; match / boundary differs / split-merged / not in record; corner RMSE, CE90 |
| Field | `backend/field.py`, `frontend/field.html` | Phone page: queue by distance, verdicts, notes, photos, GPS-checked on-site visits, boundary corners into the GNSS layer |
| Survey store | `backend/survey.py` | Tiles, overlays, layers, issues, edit log, undo history, rename, soft delete |
| Export | `backend/export.py` | GeoPackage / Shapefile in the survey's UTM zone, GeoJSON |

## Models (held-out results they were accepted on)

- **Aerial ORI + DSM** — trained on ISPRS Potsdam (`training/potsdam_unet`).
  Potsdam test tiles: mean IoU 0.723 with height, 0.721 colour only; building
  IoU 0.913.
- **Indian drone imagery** — the Potsdam model fine-tuned on Singh Nagar,
  Vijayawada (OpenAerialMap drone ORI) with weak labels from Microsoft and
  Google open building footprints plus OpenStreetMap roads
  (`training/vijayawada_finetune`). Held-out block: building IoU 0.34 → 0.77,
  footprints found 13 → 36 of 49. Known limits: adjacent houses are often
  merged; vegetation is under-detected.

The weak labels are incomplete, so pixel metrics are indicative; every model
is also checked visually on the held-out block before it is used.

## Layout

```text
backend/    Flask app + pipeline modules (above)
frontend/   workbench.html (+ records.js, fieldpanel.js, history.js), upload.html, field.html
training/   Kaggle scripts: potsdam_unet/, vijayawada_finetune/ (prepare.py builds the fine-tune set)
models/     checkpoints + their metrics (gitignored)
data/       surveys/, intake/, surveys_trash/ (gitignored)
```
