"""Stack v3: add land cover's building map to the stacked roof model.

    R    roofs v1 (frozen)
    D+   8-ch stack from the teammate bake-off (frozen): RGB + our 3 U-Net maps +
         teammate Inria roof map + teammate UAVid building map
    D++  9-ch stack: D+ inputs + land cover v1's building probability, trained on the
         Gandhinagar learning side exactly like the others
    FL   no training: R's houses plus land-cover building regions R missed, split
         into houses

Land cover v1 learned roofs from 44 countries, so it finds red-tile, dark and
apartment roofs that roofs v1 (one Gandhinagar sector) misses; the stack decides how
much to trust it. All scored on the fair Gandhinagar exam, then drawn on Indian cities.
"""
import glob, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/kaggle/working")
import bakeoff_friend as bf  # noqa: E402  (copies compare_fair/train from roofs v1 on import)
cf, log, DEV = bf.cf, bf.log, bf.DEV
import segmentation_models_pytorch as smp  # noqa: E402

WORK = Path("/kaggle/working")
STEPS = int(os.environ.get("S2_STEPS", 1500))
LC_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
LC_STD = np.array([0.229, 0.224, 0.225], np.float32)
CITIES = bf.CITIES + [("Kukatpally, Hyderabad", 17.4933, 78.3996)]


def load_landcover():
    p = glob.glob("/kaggle/input/**/landcover_oem.pt", recursive=True)[0]
    ck = torch.load(p, map_location="cpu", weights_only=False)
    net = smp.Unet("resnet50", encoder_weights=None, in_channels=3, classes=9)
    net.load_state_dict(ck["state_dict"])
    log("land cover v1:", p)
    return net.to(DEV).eval()


@torch.no_grad()
def lc_building_prob(net, rgb, tile=512, stride=384):
    h, w = rgb.shape[:2]
    im = np.pad(rgb, ((0, max(0, tile - h)), (0, max(0, tile - w)), (0, 0)), mode="reflect")
    H, W = im.shape[:2]
    x = torch.from_numpy(((im.astype(np.float32) / 255 - LC_MEAN) / LC_STD).transpose(2, 0, 1)[None]).to(DEV)
    acc = torch.zeros((H, W), device=DEV); cnt = torch.zeros((H, W), device=DEV)
    ys = list(range(0, H - tile + 1, stride)) + ([H - tile] if (H - tile) % stride else [])
    xs = list(range(0, W - tile + 1, stride)) + ([W - tile] if (W - tile) % stride else [])
    for y in ys:
        for xx in xs:
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x[:, :, y:y + tile, xx:xx + tile]).float()
            o[:, 0] = -1e4                                  # "unlabelled" is not a real class
            acc[y:y + tile, xx:xx + tile] += o.softmax(1)[0, 8]; cnt[y:y + tile, xx:xx + tile] += 1
    return (acc / cnt)[:h, :w].cpu().numpy()


def main():
    tr_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/train/*.npz", recursive=True))
    te_files = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/test/*.npz", recursive=True))
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    dplus_w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_dplus.pt", recursive=True)
    log(f"train {len(tr_files)} test {len(te_files)} | D+ weights {dplus_w}")
    if len(tr_files) < 10 or not dplus_w:
        raise SystemExit("missing inputs")
    inria, uavid, lc = bf.load_friend(fr / "unet_inria_best.pt"), bf.load_friend(fr / "uavid_unet_best.pt"), load_landcover()
    nets = [cf.load_unet(bf.ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    r_model = cf.build_maskrcnn(6); r_model.load_state_dict(torch.load(bf.ROOFS / "maskrcnn_stacked.pt", map_location="cpu")); r_model.eval()
    d8 = cf.build_maskrcnn(8); d8.load_state_dict(torch.load(dplus_w[0], map_location="cpu")); d8.eval()
    dirs = {k: Path(f"/kaggle/tmp/{k}") for k in ("s3", "s5", "s6")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def maps(rgb, which):
        ps, pe, pi, _ = cf.unet_outputs(which, rgb)
        m3 = cf.maps_u8(ps, pe, pi)
        f = (bf.friend_prob(inria, rgb) * 255).astype(np.uint8)
        u = (bf.friend_prob(uavid, rgb, cls=1) * 255).astype(np.uint8)
        lb = lc_building_prob(lc, rgb)
        m5 = np.dstack([m3, f, u]); m6 = np.dstack([m5, (lb * 255).astype(np.uint8)])
        return m3, m5, m6, lb

    def save(stem, m3, m5, m6):
        np.save(dirs["s3"] / f"{stem}.npy", m3); np.save(dirs["s5"] / f"{stem}.npy", m5); np.save(dirs["s6"] / f"{stem}.npy", m6)

    xs = np.array([int(f.stem.split("_")[2]) for f in tr_files]); mid = np.median(xs)
    for f, x in zip(tr_files, xs):
        save(f.stem, *maps(np.load(f)["rgb"], [nets[1] if x < mid else nets[0]])[:3])
    lbs = {}
    for f in te_files:
        m3, m5, m6, lb = maps(np.load(f)["rgb"], nets); save(f.stem, m3, m5, m6); lbs[f.stem] = lb
    log("maps ready")
    bf.train_stack(tr_files[:2], dirs["s6"], 9, 2)
    log("preflight OK")
    d9 = bf.train_stack(tr_files, dirs["s6"], 9, STEPS)
    torch.save(d9.state_dict(), WORK / "maskrcnn_stacked9_landcover.pt")

    def fl(r_lab, lb):
        return bf.fuse(r_lab, lb, lb)

    names = ("R  roofs v1", "D+ 8-ch (ours + teammate)", "D++ 9-ch (+ land cover)", "FL roofs v1 + land-cover fill")
    rows = {k: [] for k in names}
    for f in te_files:
        gt = np.load(f)["inst"].astype(np.int32)
        R, _ = cf.maskrcnn_labels(r_model, f, dirs["s3"], True)
        D8, _ = cf.maskrcnn_labels(d8, f, dirs["s5"], True)
        D9, _ = cf.maskrcnn_labels(d9, f, dirs["s6"], True)
        for k, lab in zip(names, (R, D8, D9, fl(R, lbs[f.stem]))):
            rows[k].append(cf.score_tile(lab, gt))
    report = {"gandhinagar_exam": {k: cf.summarise(v) for k, v in rows.items()}}
    log("EXAM\n" + json.dumps(report["gandhinagar_exam"], indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = WORK / "stack9"; out.mkdir(exist_ok=True)
    tmp = Path("/kaggle/tmp/vis9"); tmp.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, la, lo in CITIES:
        rgb = bf.fetch(la, lo)
        m3, m5, m6, lb = maps(rgb, nets)
        stem = "".join(c for c in name if c.isalnum())[:30]
        save(stem, m3, m5, m6)
        np.savez(tmp / f"{stem}.npz", rgb=rgb, inst=np.zeros(rgb.shape[:2], np.int16))
        R, _ = cf.maskrcnn_labels(r_model, tmp / f"{stem}.npz", dirs["s3"], True)
        D8, _ = cf.maskrcnn_labels(d8, tmp / f"{stem}.npz", dirs["s5"], True)
        D9, _ = cf.maskrcnn_labels(d9, tmp / f"{stem}.npz", dirs["s6"], True)
        FL = fl(R, lb)
        fig, ax = plt.subplots(1, 5, figsize=(30, 6.6))
        ax[0].imshow(rgb); ax[0].set_title(name, fontsize=13)
        counts[name] = {}
        for c, (lab, t) in enumerate(zip((R, D8, D9, FL), ("R roofs v1", "D+ 8-ch", "D++ 9-ch (+land cover)", "FL v1 + land-cover fill")), start=1):
            vis, title = bf.draw(rgb, lab, t)
            counts[name][t] = int(title.rsplit(": ", 1)[1])
            ax[c].imshow(vis); ax[c].set_title(title + " roofs", fontsize=13)
        for a in ax: a.set_xticks([]); a.set_yticks([])
        fig.tight_layout(); fig.savefig(out / f"{stem}.jpg", dpi=60); plt.close(fig)
        log(name, counts[name])
    report["city_roof_counts"] = counts
    (WORK / "stack9_report.json").write_text(json.dumps(report, indent=2))
    log("done")


if __name__ == "__main__":
    main()
