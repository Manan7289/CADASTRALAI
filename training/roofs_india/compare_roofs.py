"""Old roof model (D+) vs the India fine-tune, side by side, for a by-eye check.

The fine-tune finds far more Indian houses but also draws many more shapes that match no labelled
house (India 8 -> 64, Bhopal 45 -> 483). Those are either false alarms or real roofs the labels
lack, and the numbers cannot tell which. This draws both models on the held-out crops (labels in
yellow; predictions green if they match a house, orange if they sit mostly on a labelled building
(a piece of one), red if on unlabelled ground: an unlabelled roof or a false alarm, told apart by
eye) and on the six survey images (no labels; predictions blue). Inputs: the fine-tune kernel output, D+, the friend
models, roofs-v1, cadastraai-india-roofs, the UAVPal prep output and cadastraai-demo-rgb.
"""
import glob, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, "/kaggle/working")
import bakeoff_friend as bf  # noqa: E402
cf, log, DEV = bf.cf, bf.log, bf.DEV

WORK = Path("/kaggle/working"); VIS = WORK / "vis"; VIS.mkdir(exist_ok=True)
T = 512


def main():
    old_w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_dplus.pt", recursive=True)[0]
    new_w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_india.pt", recursive=True)[0]
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    inria, uavid = bf.load_friend(fr / "unet_inria_best.pt"), bf.load_friend(fr / "uavid_unet_best.pt")
    nets = [cf.load_unet(bf.ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    models = {}
    for name, w in (("old", old_w), ("new", new_w)):
        m = cf.build_maskrcnn(8); m.load_state_dict(torch.load(w, map_location="cpu")); m.eval(); models[name] = m
    log("models", old_w, new_w)

    def maps(rgb):
        ps, pe, pi, _ = cf.unet_outputs(nets, rgb)
        return np.dstack([cf.maps_u8(ps, pe, pi), (bf.friend_prob(inria, rgb) * 255).astype(np.uint8),
                          (bf.friend_prob(uavid, rgb, cls=1) * 255).astype(np.uint8)])

    @torch.no_grad()
    def predict(m, rgb, mp, thr=0.5):
        x = torch.from_numpy(np.concatenate([rgb / 255.0, mp / 255.0], -1).astype(np.float32).transpose(2, 0, 1))
        m.transform.min_size, m.transform.max_size = (x.shape[1],), x.shape[1]
        out = m([x.to(DEV)])[0]
        lab = np.zeros(rgb.shape[:2], np.int32)
        keep = out["scores"] >= thr
        for i, mk in enumerate((out["masks"][keep][:, 0] > 0.5).cpu().numpy(), start=1):
            lab[mk & (lab == 0)] = i
        return lab

    def tiled(m, rgb):
        """Whole survey image in 512 tiles (padded), ids kept unique."""
        H, W = rgb.shape[:2]; Hp, Wp = -(-H // T) * T, -(-W // T) * T
        pad = np.zeros((Hp, Wp, 3), np.uint8); pad[:H, :W] = rgb
        lab = np.zeros((Hp, Wp), np.int32); n = 0
        for y in range(0, Hp, T):
            for x in range(0, Wp, T):
                t = pad[y:y + T, x:x + T]; l = predict(m, t, maps(t))
                lab[y:y + T, x:x + T] = np.where(l > 0, l + n, 0); n += int(l.max())
        return lab[:H, :W]

    def sort_unmatched(pred, gt):
        """Each predicted shape: matched (IoU>=0.5 with a label), part (>=50% on labelled building), or off-label."""
        kinds = {}
        for v, sl in enumerate(ndi.find_objects(pred), start=1):
            if sl is None:
                continue
            mk = pred[sl] == v; g = gt[sl][mk]
            ids, cnt = np.unique(g[g > 0], return_counts=True)
            best = 0.0
            for i, c in zip(ids, cnt):
                best = max(best, c / (mk.sum() + (gt == i).sum() - c))
            kinds[v] = "matched" if best >= 0.5 else ("part" if (g > 0).mean() >= 0.5 else "off")
        return kinds

    def draw(rgb, pred, gt=None, kinds=None):
        im = rgb.copy()
        if gt is not None:
            e = (ndi.maximum_filter(gt, 3) != ndi.minimum_filter(gt, 3)) & (ndi.maximum_filter(gt, 3) > 0)
            im[e] = (255, 255, 0)
        col = {"matched": (0, 220, 0), "part": (255, 140, 0), "off": (255, 0, 0), None: (0, 200, 255)}
        pe = (ndi.maximum_filter(pred, 3) != ndi.minimum_filter(pred, 3)) & (ndi.maximum_filter(pred, 3) > 0)
        for v in np.unique(pred[pe]):
            if v:
                im[pe & (ndi.maximum_filter(pred, 3) == v)] = col[(kinds or {}).get(v)]
        return im

    summary = {}
    crops = sorted(glob.glob("/kaggle/input/**/intest_*.npz", recursive=True)) + sorted(glob.glob("/kaggle/input/**/uptest_*.npz", recursive=True))
    for f in crops:
        d = np.load(f); rgb, gt = d["rgb"], d["inst"].astype(np.int32); mp = maps(rgb)
        row, pics = {"labelled_houses": int(len(np.unique(gt)) - 1)}, []
        for name, m in models.items():
            p = predict(m, rgb, mp); k = sort_unmatched(p, gt)
            row[name] = {c: sum(1 for x in k.values() if x == c) for c in ("matched", "part", "off")}
            pics.append(draw(rgb, p, gt, k))
        summary[Path(f).stem] = row
        cv2.imwrite(str(VIS / f"crop_{Path(f).stem}.jpg"), cv2.cvtColor(np.hstack(pics), cv2.COLOR_RGB2BGR))
        log(Path(f).stem, json.dumps(row))
    tot = {n: {c: sum(r[n][c] for r in summary.values()) for c in ("matched", "part", "off")} for n in models}
    log("TOTAL", json.dumps(tot))

    for f in sorted(glob.glob("/kaggle/input/**/cadastraai-demo-rgb/**/*.npz", recursive=True)):
        d = np.load(f); rgb = d["rgb"]; valid = d["valid"] if "valid" in d else np.ones(rgb.shape[:2], bool)
        pics, cnt = [], {}
        for name, m in models.items():
            p = tiled(m, rgb); p[~valid] = 0; cnt[name] = int(len(np.unique(p)) - 1); pics.append(draw(rgb, p))
        summary[Path(f).stem] = {"houses": cnt}
        im = np.hstack(pics); s = min(1.0, 2400 / im.shape[1])
        cv2.imwrite(str(VIS / f"survey_{Path(f).stem}.jpg"), cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
        log(Path(f).stem, cnt)
    (WORK / "compare_roofs.json").write_text(json.dumps({"total": tot, "per": summary}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
