"""Fine-tune the approved roof stack (D+) on hand-labelled Indian roofs.

D+ is an 8-channel Mask R-CNN: RGB + our U-Net maps (3) + the teammate's Inria roof map + his
UAVid building map, trained on one Gandhinagar sector. Outside that sector it finds few roofs
(Dwarka 9, Chandigarh 4), so the app fills in the rest from land cover, more roughly.

This continues training D+ from its weights on Gandhinagar + hand-labelled roofs from Jaipur,
HSR Layout, Dwarka, Chandigarh and Singh Nagar (training/roofs_india/labels; half of every batch
Indian), with the same crop sampler and 8-channel maps as D+. Scored before and after on
held-out Indian crops and on the fair Gandhinagar exam, so a gain in India that costs Gandhinagar
shows up. Inputs: roofs-v1, friend models, the D+ kernel output, the Gandhinagar mosaic kernel
output and the dataset cadastraai-india-roofs.
"""
import glob, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/kaggle/working")
import bakeoff_friend as bf  # noqa: E402
cf, log, DEV = bf.cf, bf.log, bf.DEV

WORK = Path("/kaggle/working")
STEPS = int(os.environ.get("STEPS", 2000))
INDIA_SHARE = 0.5


def main():
    tr = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/train/*.npz", recursive=True))
    te = sorted(Path(p) for p in glob.glob("/kaggle/input/**/gn_mosaic/test/*.npz", recursive=True))
    itr = sorted(Path(p) for p in glob.glob("/kaggle/input/**/intrain_*.npz", recursive=True))
    ite = sorted(Path(p) for p in glob.glob("/kaggle/input/**/intest_*.npz", recursive=True))
    d8w = glob.glob("/kaggle/input/**/maskrcnn_stacked8_dplus.pt", recursive=True)[0]
    fr = Path(glob.glob("/kaggle/input/**/unet_inria_best.pt", recursive=True)[0]).parent
    log(f"gandhinagar train {len(tr)} test {len(te)} | india train {len(itr)} test {len(ite)} | D+ {d8w}")
    inria, uavid = bf.load_friend(fr / "unet_inria_best.pt"), bf.load_friend(fr / "uavid_unet_best.pt")
    nets = [cf.load_unet(bf.ROOFS / f"s1_fold{k}.pt") for k in (0, 1)]
    s5 = Path("/kaggle/tmp/s5"); s5.mkdir(parents=True, exist_ok=True)

    def maps(rgb, which):
        ps, pe, pi, _ = cf.unet_outputs(which, rgb)
        return np.dstack([cf.maps_u8(ps, pe, pi), (bf.friend_prob(inria, rgb) * 255).astype(np.uint8),
                          (bf.friend_prob(uavid, rgb, cls=1) * 255).astype(np.uint8)])

    xs = np.array([int(f.stem.split("_")[2]) for f in tr]); mid = np.median(xs)
    for f, x in zip(tr, xs):                       # out-of-fold maps, exactly as D+ was trained
        np.save(s5 / (f.stem + ".npy"), maps(np.load(f)["rgb"], [nets[1] if x < mid else nets[0]]))
    for f in te + itr + ite:
        np.save(s5 / (f.stem + ".npy"), maps(np.load(f)["rgb"], nets))
    log("maps ready")

    def evaluate(m, files):
        rows = []
        for f in files:
            lab, gt = cf.maskrcnn_labels(m, f, s5, True)
            rows.append(cf.score_tile(lab, gt.astype(np.int32)))
        return cf.summarise(rows)

    m = cf.build_maskrcnn(8); m.load_state_dict(torch.load(d8w, map_location="cpu")); m.eval()
    before = {"india": evaluate(m, ite), "gandhinagar": evaluate(m, te)}
    log("BEFORE", json.dumps(before))

    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(STEPS, 1))
    scaler = torch.amp.GradScaler()
    for step in range(STEPS):
        batch = [cf.sample(random.choice(itr if i < 4 * INDIA_SHARE else tr), s5, True) for i in range(4)]
        imgs = [b[0].to(DEV) for b in batch]
        tgts = [{k: v.to(DEV) for k, v in b[1].items()} for b in batch]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(imgs, tgts).values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if step % 200 == 0:
            log(f"step {step}/{STEPS} loss {loss.item():.3f}")
    m.eval()
    after = {"india": evaluate(m, ite), "gandhinagar": evaluate(m, te)}
    log("AFTER", json.dumps(after))
    torch.save(m.state_dict(), WORK / "maskrcnn_stacked8_india.pt")
    (WORK / "roofs_india_metrics.json").write_text(json.dumps({"before": before, "after": after, "steps": STEPS,
                                                               "india_train_crops": len(itr), "india_test_crops": len(ite)}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
