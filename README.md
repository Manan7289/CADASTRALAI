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
3. **Real land-parcel delineation** (`backend/parcels.py`): OpenStreetMap
   doesn't carry land-ownership boundaries (that's government cadastral
   data, not a public API), so parcels are inferred the way the UAV-cadastral
   literature this project's own submission cites does it — a Voronoi
   tessellation around each real building position, clipped by the real
   road/rail/water network. Every parcel gets real area/perimeter, a
   land-use class (from real OSM `landuse` tags or inferred), and a
   road-frontage check.
4. **A real rule engine** (`backend/rules.py`): shapely buffers and
   intersections — not per-parcel hand-scripting — check every extracted
   footprint against the real rail line (30 m buffer), real water features
   (15 m buffer), real government-tagged land, and real road network.
5. **Two dashboards** sharing the same data (`frontend/index.html` and
   `frontend/detection.html`, served by `backend/app.py`):
   - **Cadastral Map** (`index.html`) — the parcel layer: click a plot for
     its dimensions, land-use, and road connectivity.
   - **Compliance & Detection** (`detection.html`) — the building layer:
     toggle AI-extracted vs. official OSM, click a footprint for its rule
     violations, generate a field verification note.
6. **Upload your own area** (`frontend/upload.html` + `backend/upload_pipeline.py`)
   — the same detect → delineate → check pipeline, but on a drone image or
   video you supply, for wherever you say it was taken. See "Uploading your
   own imagery" below for exactly what changes.

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
  numbers (rerun `model.py` and check `frontend/data/metrics.json` for the
  exact current values — they shift slightly whenever the AOI or training
  data changes) are real generalization metrics, not measured against
  incomplete ground truth.
- The trained model is then applied to the full AOI. Most of its predictions
  there don't match an OSM building — that's flagged as `UNRECORDED` and is
  the actual point: the model finds real structures the official layer is
  missing, exactly the encroachment/unrecorded-construction signal a
  cadastral tool needs, without requiring a second drone flight to get it.

This RandomForest pipeline is a genuine, honestly-evaluated baseline, and it
stays in the repo as the zero-dependency path. It is no longer the best
detector here, though — see the next section.

## The deep-learning pipeline (`backend/ml/`)

The RandomForest's ceiling is structural, not a tuning problem: it can only
learn from the ~50 OSM footprints that happen to exist in this one AOI, so it
can never be better than that label set, and it cannot move to a new town at
all. The fix is to stop borrowing labels from the target area and train on
datasets that ship real, dense, hand-digitised ground truth.

| Model | Trained on | Used for |
| --- | --- | --- |
| `inria_unet.pt` | [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) — 0.3 m/px nadir ortho, binary building masks, 5 cities | The Igatpuri map, and nadir uploads |
| `uavid_unet.pt` | [UAVid 2020](https://uavid.nl/) — oblique 50 m-altitude drone video, 8 semantic classes | The Upload page, which is what actually receives drone footage |

Both are U-Nets with an ImageNet-pretrained ResNet-34 encoder, trained on this
machine's own GPU. They are deliberately two models rather than one: a model
trained on oblique low-altitude video produces nonsense on nadir orthoimagery,
and vice versa.

### The resolution problem, which is the whole ballgame

A CNN keys on *how many pixels a building occupies*, so the ground sample
distance of the input has to match what it trained on. Inria is 0.3 m/px. The
AOI mosaic was zoom 17 — **1.12 m/px**, about 4x too coarse, on which the model
predicts essentially nothing useful.

Probing Esri's tile service for this AOI shows it carries real imagery only to
**zoom 18 (0.562 m/px)**; zoom 19 and 20 return blank grey placeholder tiles.
So the gap is closed from both ends:

- `fetch_imagery.py --zoom 18 --prefix aoi_image_z18` fetches the sharpest real
  imagery that exists here, into its own file (the zoom-17 mosaic stays as the
  web backdrop — a 5000 px PNG has no business being shipped to a browser).
- Inference resamples that up to 0.3 m/px before running the network.
- Training degrades patches the same way (downscale, then back up), so the model
  is trained on exactly the soft, upsampled input it receives at inference.

That last augmentation is the single change that matters most for this transfer;
without it the upsampled AOI looks like nothing in the training set.

### Running it

```bash
cd backend
pip install -r requirements.txt        # includes torch + segmentation-models-pytorch

python fetch_imagery.py --zoom 18 --prefix aoi_image_z18   # sharpest real imagery for this AOI
python -m ml.prepare_inria --tiles-per-city 8              # ~3 GB from HuggingFace, then tiles it
python -m ml.train_inria --epochs 25 --batch 8             # GPU; metrics -> data/models/inria_metrics.json
python -m ml.infer_buildings                               # mask + overlay + extracted_buildings.geojson
python -m ml.compare_baseline                              # U-Net vs RandomForest on the same ground truth

python -m ml.prepare_uavid --frame-stride 2                # ~2 GB; every 2nd frame (sequences repeat)
python -m ml.train_uavid --epochs 30 --batch 8
python -m ml.infer_semantic --image <a drone photo>        # the 8-class colour map
```

`ml.infer_buildings` writes `extracted_buildings.geojson` in exactly the schema
`model.py` wrote, so `parcels.py`, `rules.py`, `build_static.py` and the whole
frontend are untouched by the swap. Run `build_static.py` afterwards as usual;
it picks up the U-Net mask as a toggleable map layer if it is there, and builds
fine without it if it is not.

Neither dataset needs a manual registration step: Inria comes from the
`blanchon/INRIA-Aerial-Image-Labeling` mirror and UAVid from
`dronefreak/UAVid-2020`, both public on HuggingFace. Both are research datasets
(UAVid is CC BY-NC-SA) — fine for this submission, worth checking before any
commercial use.

### Reading the numbers honestly

`ml.compare_baseline` scores both detectors against the same verified colony,
with one asymmetry that must be stated whenever the table is shown: **the
RandomForest was trained on those very buildings**, so its score there is a best
case it cannot reproduce anywhere else, while the U-Net has never seen Igatpuri
or India at all. A U-Net number merely *close* to the RandomForest's is
therefore the decisively better result, because only one of the two can be
taken to a new district.

Exact current values live in `data/processed/detector_comparison.json` and
`data/models/*_metrics.json` rather than being copied here, since they shift
whenever the AOI or training set changes.

## Running it locally

```bash
cd backend
pip install -r requirements.txt
python fetch_data.py       # ~1 min, needs internet (Overpass API)
python fetch_imagery.py    # ~1 min, needs internet (Esri tile service)
python model.py            # ~10s, trains + predicts, no internet needed
python build_static.py     # runs the rule engine once, bakes output into frontend/data/
python app.py              # serves http://127.0.0.1:5050
```

Open `http://127.0.0.1:5050` for the Cadastral Map, or
`http://127.0.0.1:5050/detection.html` for Compliance & Detection (there's a
nav link between them). `app.py` is a thin Flask server that just
serves `frontend/` (including `frontend/data/`) — it's the same static
files Vercel deploys, so what you see locally is exactly what production
looks like. Only the two `fetch_*` scripts need a network; everything from
`model.py` onward runs fully offline.

Changed the model or a rule and want to see it locally? Rerun `model.py`
and/or `build_static.py`, then just refresh the browser — no server
restart needed, since `app.py` reads `frontend/data/` fresh on every
request.

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

## Uploading your own imagery

Open `http://127.0.0.1:5050/upload.html` (Flask must be running -- this
needs a real Python process to run OpenCV and live Overpass fetches on
demand, so **it does not work on the static Vercel deployment**). Give it a
drone photo, a satellite orthoimage, or a short video (its middle frame is
used -- this is a single frame, not multi-frame photogrammetric stitching),
plus a centre latitude/longitude and an approximate ground width in metres
(no EXIF/GPS parsing is attempted, so this is how it's georeferenced). It
takes roughly 15-40 seconds:

- Road/rail/water/government/land-use layers are fetched live from
  OpenStreetMap for that location -- real data, same as the demo AOI, just
  for wherever you pointed it.
- Building detection falls back to an unsupervised Otsu brightness/texture
  threshold instead of the trained RandomForest, because there's no labelled
  ground truth for an arbitrary new area to train on. It's calibrated to
  plausible real building sizes (20-3,000 m²) rather than a fixed pixel
  count, but it is less accurate than the curated demo AOI -- the sidebar
  says so explicitly rather than presenting a fake accuracy figure.
- The same parcel delineation (`parcels.build_parcels_from_data`) and rule
  engine (`rules.evaluate_layers`) run on the result, unchanged.

Results land at `index.html?session=<id>` and `detection.html?session=<id>`
-- the `?session=` param tells the frontend to fetch from
`/uploads/<id>/...` (this upload's own generated files) instead of the
fixed demo bundle under `data/`.

## Project layout

```text
backend/
  fetch_data.py      real OSM vectors (buildings, roads, rail, water, govt land)
  fetch_imagery.py   real satellite tile download + stitching + georeferencing
  geo_utils.py       lon/lat <-> local-metre projection (buffers need real metres, not degrees)
  model.py           the RandomForest training/inference/evaluation pipeline
  parcels.py         Voronoi-based land-parcel delineation + attributes
  rules.py           the shapely-based compliance rule engine
  app.py             Flask API + static frontend server + upload endpoint (local dev only)
  upload_pipeline.py runs detection/parcels/rules on a user-uploaded image or video
  build_static.py    bakes parcels + rules + reports into frontend/data/ for static hosting
frontend/
  index.html         Cadastral Map page (parcels, land-use, dimensions)
  detection.html     Compliance & Detection page (buildings, AI vs OSM, alerts)
  upload.html        upload form for processing your own drone image/video
  shared.js          map setup + icon/color helpers + ?session= data routing, used by all pages
  app.css            shared styling for all pages
  data/              generated by build_static.py -- what Vercel actually serves
data/
  processed/         everything fetch_*.py and model.py produce (GeoJSON, mosaic, metrics)
  uploads/           per-upload generated results (gitignored, local only)
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
- Real cadastral boundaries later: if a government parcel layer (Bhu Naksha /
  SVAMITVA) ever becomes available for an AOI, swap it in for `parcels.py`'s
  Voronoi output — `parcels.geojson`'s schema (area, perimeter, landuse,
  road_connected, building_ids) is what the Cadastral Map page actually
  consumes, not how it was produced.
