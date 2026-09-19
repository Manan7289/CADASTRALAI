"""Run the approved models on any prepared survey image and write one bundle per image.

This is the GPU half of "New Survey" (and of reprocessing old surveys): the app warps the
upload to its UTM zone at 0.3 m (backend/kaggle_jobs.py), uploads it as the private dataset
cadastraai-job-inputs, and this kernel runs:

    roofs       D+ stack: our U-Net (2 folds) + teammate Inria and UAVid maps -> 8-channel Mask R-CNN
    land cover  SegFormer-B2 on OpenEarthMap (land cover v2)

Input:  /kaggle/input/**/job_*.npz  with rgb uint8 HxWx3, valid bool, info json
Output: /kaggle/working/bundles/<same stem>.npz, the bundle format backend/import_bundle.py reads.
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/kaggle/working")
import bakeoff_friend as bf  # noqa: E402  (loads roofs v1 helpers on import)
import demo_bundles as db    # noqa: E402  (land cover loader and tiled inference)
cf, log = bf.cf, bf.log

WORK = Path("/kaggle/working")


def main():
    jobs = sorted(glob.glob("/kaggle/input/**/job_*.npz", recursive=True))
    log(f"{len(jobs)} input image(s)")
    if not jobs:
        raise SystemExit("no job_*.npz inputs attached")
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    d8w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_dplus.pt", recursive=True)[0]
    inria, uavid = bf.load_friend(fr / "unet_inria_best.pt"), bf.load_friend(fr / "uavid_unet_best.pt")
    nets = [cf.load_unet(bf.ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    d8 = cf.build_maskrcnn(8); d8.load_state_dict(torch.load(d8w, map_location="cpu")); d8.eval()
    lc = db.load_landcover()
    out = WORK / "bundles"; out.mkdir(exist_ok=True)
    tmp = Path("/kaggle/tmp/p"); tmp.mkdir(parents=True, exist_ok=True)
    for path in jobs:
        stem = Path(path).stem
        j = np.load(path, allow_pickle=False)
        rgb, valid, info = j["rgb"], j["valid"].astype(bool), json.loads(str(j["info"]))
        rgb = rgb.copy(); rgb[~valid] = 0
        ps, pe, pi, _ = cf.unet_outputs(nets, rgb)
        m5 = np.dstack([cf.maps_u8(ps, pe, pi), (bf.friend_prob(inria, rgb) * 255).astype(np.uint8),
                        (bf.friend_prob(uavid, rgb, cls=1) * 255).astype(np.uint8)])
        np.save(tmp / f"{stem}.npy", m5)
        np.savez(tmp / f"{stem}.npz", rgb=rgb, inst=np.zeros(rgb.shape[:2], np.int16))
        roofs, _ = cf.maskrcnn_labels(d8, tmp / f"{stem}.npz", tmp, True)
        roofs[~valid] = 0
        lcp = db.lc_probs(lc, rgb) if min(rgb.shape[:2]) >= 512 else _lc_small(lc, rgb)
        info["models"] = {"roofs": "D+ stack: U-Net (roofs v1 folds) + teammate Inria/UAVid maps -> 8-ch Mask R-CNN",
                          "land_cover": "SegFormer-B2, OpenEarthMap (land cover v2)"}
        np.savez_compressed(out / f"{stem}.npz", rgb=rgb, valid=valid, roofs=roofs.astype(np.int32),
                            lc_probs=np.clip(lcp * 255, 0, 255).astype(np.uint8), info=json.dumps(info))
        log(info.get("name"), "| grid", rgb.shape, "| roofs", len(np.unique(roofs)) - 1)
    log("done")


def _lc_small(net, rgb):
    """images smaller than one 512 tile: pad, run, crop"""
    h, w = rgb.shape[:2]
    pad = np.pad(rgb, ((0, max(0, 512 - h)), (0, max(0, 512 - w)), (0, 0)), mode="reflect")
    return db.lc_probs(net, pad)[:, :h, :w]


if __name__ == "__main__":
    main()
