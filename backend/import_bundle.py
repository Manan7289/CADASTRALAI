"""Import a model bundle (made on Kaggle by training/deploy/demo_bundles.py) as a survey.

The heavy models run on Kaggle; this only does the light part locally: parcels grown
from the roof model's houses, the 8-class land-cover layer, tiles and the survey folder.

    python import_bundle.py <bundle.npz> [<bundle.npz> ...]
"""
import json
import sys

import numpy as np
from affine import Affine
from rasterio.crs import CRS

import parcel_extract
import roof_fill
import survey

# land cover v2 (OpenEarthMap) -> the app's 5 segmentation classes
# (0 clutter, 1 building, 2 road, 3 low vegetation, 4 tree) used for corridors and parcels
LC_TO_APP = {1: 0, 2: 3, 3: 0, 4: 2, 5: 4, 6: 0, 7: 3, 8: 1}

MODEL_INFO = {
    "key": "stack_dplus_landcover_v2",
    "name": "Roofs: stacked ensemble (our U-Net + Mask R-CNN, teammate Inria + UAVid maps); "
            "land cover: SegFormer-B2 (OpenEarthMap)",
    "summary": "Roofs the stack misses but land cover marks as building are filled in and tagged for review. "
               "Roofs, fair Gandhinagar exam: 88% of houses found, 12.5% of touching pairs merged, outline IoU 0.89. "
               "Land cover, OpenEarthMap validation: mIoU 0.67 (road 0.65, tree 0.71, grass 0.58, bare land 0.44).",
    "limits": "Roofs trained on one planned Indian sector plus WHU; misses some red-tile / dark roofs and very "
              "dense blocks. Land cover weakest on bare land (IoU 0.44). Satellite imagery ~0.3 m, not drone.",
}


def build_survey(name, source, loaded, lcp, model_info, roofs=None):
    """Survey from land cover v2 probabilities (9,H,W) and, if there is one, a roof-instance raster
    from the stacked roof model. Houses are split out of the land-cover building map where the roof
    model has none (everywhere, without a roof model). Used by bundle import and by New Survey uploads."""
    rgb, valid = loaded["rgb"], loaded["valid"]
    gsd = abs(loaded["transform"].a)
    probs5 = np.zeros((5,) + rgb.shape[:2], np.float32)
    for k, a in LC_TO_APP.items():
        probs5[a] += lcp[k]
    probs5 /= np.clip(probs5.sum(0, keepdims=True), 1e-6, None)
    landcover = (lcp[1:].argmax(0) + 1).astype(np.uint8)
    landcover[~valid] = 0
    has_roof_model = roofs is not None
    roofs = np.zeros(rgb.shape[:2], np.int32) if roofs is None else roofs
    roofs, filled = roof_fill.fill_missed_roofs(roofs, lcp[8], valid, gsd)
    extracted = parcel_extract.extract(probs5, rgb, loaded["transform"], ndsm=loaded.get("ndsm"), valid=valid,
                                       inst_override=roofs, landcover=landcover,
                                       inst_fill=filled if has_roof_model else None, paved=landcover == 3,
                                       building_source="roof model" if has_roof_model else "land cover model")
    meta = survey.create(name, source, loaded, probs5, extracted, model_info, landcover=landcover)
    # last tidy-up of hairline artefacts (specks, pinched plots) with the workbench's own auto-fix
    if json.loads((survey.survey_dir(meta["id"]) / "issues.geojson").read_text())["features"]:
        survey.auto_fix(meta["id"])
    return meta


def import_bundle(path):
    b = np.load(path, allow_pickle=False)
    info = json.loads(str(b["info"]))
    rgb, valid, roofs = b["rgb"], b["valid"].astype(bool), b["roofs"].astype(np.int32)
    lcp = b["lc_probs"].astype(np.float32) / 255.0
    loaded = {"rgb": rgb, "valid": valid, "transform": Affine(*info["transform"]), "crs": CRS.from_string(info["crs"]),
              "ndsm": None, "height_source": None}
    source = f"{info['imagery']} · processed at {info['gsd_m']} m · models run on Kaggle"
    meta = build_survey(info["name"], source, loaded, lcp, MODEL_INFO, roofs=roofs)
    s = meta["stats"]
    print(f"{info['name']}: survey {meta['id']} | {s['parcels']} parcels, {s['buildings']} buildings "
          f"({s['buildings_filled']} filled in from land cover), "
          f"{s['corridor_length_m']} m corridors | land cover {s.get('land_cover_fraction')}")
    return meta


if __name__ == "__main__":
    for p in sys.argv[1:]:
        import_bundle(p)
