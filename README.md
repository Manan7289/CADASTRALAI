# CadastraAI — real pipeline

SIH 2026, PS 26012 — AI-Based Automated Urban Parcel Mapping and Cadastral
Feature Extraction from Drone Imagery. Team ELAVI.

This is not the earlier concept demo with hand-authored synthetic data. Every
input here is real: live OpenStreetMap vector data, a stitched real satellite
mosaic, a model actually trained on this machine, and a rule engine running
real shapely geometry against all of it.

## What it does

1. **Real data** for a real place (Igatpuri, Maharashtra — a railway town
   chosen because it has buildings, roads, a railway line, water and
   government land all in one small area): `backend/fetch_data.py` pulls
   live OSM vectors via Overpass; `backend/fetch_imagery.py` stitches real
   Esri World Imagery tiles into one georeferenced mosaic.
2. **A model that actually trains, on CPU, in ~6 seconds**
   (`backend/model.py`): a scikit-learn RandomForest pixel classifier learns
   what a rooftop looks like from real OSM building footprints, then extracts
   candidate building polygons across the whole AOI via classical CV
   (contour extraction on the probability map).
3. **A real rule engine** (`backend/rules.py`): shapely buffers and
   intersections — not per-parcel hand-scripting — check every extracted
   footprint against the real rail line (30 m buffer), real water features
   (15 m buffer), real government-tagged land, and real road network.
4. **A live dashboard** (`frontend/index.html`, served by `backend/app.py`):
   toggle between the AI-extracted layer and the official OSM layer, click
   any footprint for its findings, generate a field verification note.

## An honest complication (worth knowing before you present this)

OSM's building tagging in this AOI is complete only inside one densely,
regularly-spaced housing colony — the rest of the actual town has real,
visible buildings with no OSM tag at all. This was discovered by manually
overlaying the OSM mask on the satellite image (see the development history
in `model.py`'s docstring) and is a genuine, common real-world condition, not
a bug: **official digital records are frequently incomplete, which is
exactly the gap this project targets.**

Two consequences, both handled honestly rather than hidden:

- The model is trained and evaluated **only** on the one verified, completely
  digitised colony (found automatically by nearest-neighbour building
  spacing), using a genuine 5-fold random train/test split. The reported
  numbers — **~57% of held-out buildings correctly located, 0.81 ROC-AUC,
  67% precision** — are real generalization metrics, not measured against
  incomplete ground truth.
- The trained model is then applied to the full AOI. Most of its predictions
  there don't match an OSM building — that's flagged as `UNRECORDED` and is
  the actual point: the model finds real structures the official layer is
  missing, exactly the encroachment/unrecorded-construction signal a
  cadastral tool needs, without requiring a second drone flight to get it.

If asked "why isn't accuracy higher / why not use a full segmentation
network": full U-Net/Mask R-CNN training needs GPU-hours over tens of
thousands of labelled tiles (SpaceNet, WHU) — not available on a CPU laptop
overnight. This RandomForest pipeline is a genuine, honestly-evaluated
baseline demonstrating the same end-to-end architecture described in the
submission's technical approach; swapping in a properly-trained segmentation
model later only changes `model.py`'s output format, nothing downstream.

## Running it locally (with the live Flask backend)

```bash
pip install -r requirements.txt

cd backend
python fetch_data.py       # ~1 min, needs internet (Overpass API)
python fetch_imagery.py    # ~1 min, needs internet (Esri tile service)
python model.py            # ~10s, trains + predicts, no internet needed
python app.py              # serves http://127.0.0.1:5050
```

Open `http://127.0.0.1:5050`. Everything after `model.py` runs fully offline
— the satellite mosaic is a local file, Leaflet is vendored locally, and the
API is your own machine. Only the two `fetch_*` scripts need a network.

Re-running `model.py` alone (e.g. after tweaking a threshold) is enough to
refresh the map — just restart `app.py` afterwards so it drops its
in-memory rule-engine cache.

## Deploying (Vercel — static, no backend needed)

Vercel's serverless functions aren't a good fit for a RandomForest + shapely
pipeline with in-memory caching, and none of that needs to run per-request
anyway — the model and rules only change when you deliberately rerun them.
So `backend/build_static.py` runs the rule engine once and bakes everything
(alerts, precomputed verification notes, all reference layers, the mosaic)
into plain files under `frontend/data/`, which `frontend/index.html` fetches
directly. `app.py` is not used in this path at all.

```bash
cd backend
python build_static.py     # after fetch_data.py / fetch_imagery.py / model.py have already run
```

Then from the repo root:

```bash
git add -A
git commit -m "Rebuild static bundle"
git push
```

On [vercel.com](https://vercel.com) → **Add New → Project → Import** this
GitHub repo. `vercel.json` at the repo root already sets
`outputDirectory: "frontend"` with no build command, so Vercel just serves
`frontend/` as a static site — no configuration needed in the dashboard.
Every push to the connected branch redeploys automatically.

To publish a change to the map/rules: rerun the relevant script(s) locally
(`model.py` and/or `rules.py`-affecting changes), then `build_static.py`,
then commit and push — Vercel picks it up from there.

## Project layout

```text
backend/
  fetch_data.py      real OSM vectors (buildings, roads, rail, water, govt land)
  fetch_imagery.py   real satellite tile download + stitching + georeferencing
  geo_utils.py       lon/lat <-> local-metre projection (buffers need real metres, not degrees)
  model.py           the RandomForest training/inference/evaluation pipeline
  rules.py           the shapely-based compliance rule engine
  app.py             Flask API + static frontend server (local dev only)
  build_static.py    bakes rules + reports into frontend/data/ for static hosting
frontend/
  index.html         Leaflet dashboard (vendored, works offline)
  data/              generated by build_static.py -- what Vercel actually serves
data/
  processed/         everything fetch_*.py and model.py produce (GeoJSON, mosaic, metrics)
scripts/
  explore_area.py    the Overpass probe used to pick Igatpuri as the AOI
```

## Extending it (the "simple code, room for amendments" requirement)

- New rule: add one function-shaped block to `rules.py`'s `evaluate_all()` —
  it's a plain list of `if geom.intersects(...): alerts.append(...)` checks.
- New reference layer (e.g. a flood zone or CRZ boundary): add a fetcher in
  `fetch_data.py`, load it in `rules.py`'s `load_layers()`, buffer/intersect
  it like the others.
- Better model later: `model.py` only needs to keep writing
  `extracted_buildings.geojson` in the same schema — `rules.py`, `app.py` and
  the frontend don't change.
