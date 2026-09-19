"""Ensemble member 1: U-Net with RAMP's multimask + Nacala's separation tricks.

- Outputs: 4-class softmax (background / building / boundary / close_contact, the
  RAMP baseline's targets), a shrunk-roof interior (Nacala's DOW idea: neighbours'
  seeds never touch) and a distance-to-edge map.
- Loss: class-weighted cross-entropy x the U-Net/Nacala border weight map, so
  background squeezed between two houses costs up to 11x; BCE+Dice on interior;
  L1 on distance.
- Starts from Nacala's trained U-Net-DOW (ResNet34, drone roofs) where shapes match.
- Trains on RAMP South Asian chips + the Gandhinagar training area, at 2x upscale.
- Writes instances and probability maps for both test sets, for the fusion step.
"""
import json
import os
import random
import time
from pathlib import Path

import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
import segmentation_models_pytorch as smp  # noqa: E402
from common import (BG, BOUNDARY, BUILDING, CONTACT, UPSCALE, crops, distance, downscale_labels, find,
                    interior, load_gandhinagar, load_ramp, log, multimask, pack_instances, panel,
                    score_tile, summarise, unet_decode, weight_map)

WORK = Path("/kaggle/working")
EPOCHS = int(os.environ.get("EPOCHS", 14))
BUDGET_H = float(os.environ.get("BUDGET_H", 3.0))
BATCH = 12
N_OUT = 6                           # 4 multimask logits, interior, distance
INT, DIST = 4, 5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda"
random.seed(0); np.random.seed(0); torch.manual_seed(0)


class Chips(Dataset):
    def __init__(self, samples, train=True):
        self.s, self.train = samples, train

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        _, rgb, inst, partial = self.s[i]
        inst = inst.astype(np.int32)
        if self.train:
            k = np.random.randint(4)
            rgb, inst = np.rot90(rgb, k).copy(), np.rot90(inst, k).copy()
            if np.random.rand() < 0.5:
                rgb, inst = rgb[:, ::-1].copy(), inst[:, ::-1].copy()
        # targets at native scale (cheap), then upscaled with the image
        mm = multimask(inst, boundary_px=1, contact_px=2).astype(np.int64)
        it = interior(inst, shrink_px=2).astype(np.float32)
        ds = distance(inst, cap_px=6)
        wm = weight_map(inst, w0=10.0, sigma=2.5)
        if partial:
            # unlabelled houses exist: background far from any labelled house is "unknown"
            known = np.asarray(inst > 0)
            from scipy import ndimage as ndi
            known = ndi.binary_dilation(known, iterations=8)
            mm[~known] = -100
            it[~known] = -1; ds[~known] = -1
        up = lambda a: np.repeat(np.repeat(a, UPSCALE, 0), UPSCALE, 1)
        x = up(rgb).astype(np.float32) / 255.0
        if self.train:
            x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.06, 0.06), 0, 1)
        x = ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(up(mm)), torch.from_numpy(up(it))[None],
                torch.from_numpy(up(ds))[None], torch.from_numpy(up(wm)))


def build_model():
    m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=N_OUT)
    # Nacala ships the U-Net-DOW checkpoint as a file named "best_model" (no extension)
    cands = [p for p in find("nacala/unet_dow1/**/*") if Path(p).is_file()]
    log("nacala unet files:", cands)
    if cands:
        ck = torch.load(cands[0], map_location="cpu", weights_only=False)
        sd = ck
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
        own = m.state_dict(); n = 0
        for k, v in sd.items():
            k2 = k.replace("module.", "")
            if k2 in own and own[k2].shape == v.shape:
                own[k2] = v; n += 1
        m.load_state_dict(own)
        log(f"init from Nacala U-Net-DOW: {n}/{len(own)} tensors matched")
    return m.to(DEV)


def dice(logit, target):
    msk = (target >= 0).float(); p = torch.sigmoid(logit) * msk; t = target.clamp_min(0) * msk
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def loss_fn(out, mm, it, ds, wm):
    cw = torch.tensor([1.0, 1.0, 2.0, 4.0], device=out.device)
    ce = F.cross_entropy(out[:, :4], mm, weight=cw, ignore_index=-100, reduction="none")
    valid = (mm >= 0).float()
    l_ce = (ce * wm * valid).sum() / (wm * valid).sum().clamp_min(1)
    m_i = (it >= 0).float()
    l_int = (F.binary_cross_entropy_with_logits(out[:, INT:INT + 1], it.clamp_min(0), reduction="none") * m_i).sum() \
        / m_i.sum().clamp_min(1) + dice(out[:, INT:INT + 1], it)
    m_d = (ds >= 0).float()
    l_d = ((torch.sigmoid(out[:, DIST:DIST + 1]) - ds.clamp_min(0)).abs() * m_d).sum() / m_d.sum().clamp_min(1)
    return l_ce + l_int + 0.5 * l_d


@torch.no_grad()
def predict(model, rgb):
    """rgb native -> (probs dict at upscaled res, labels at native res, scores)."""
    x = np.repeat(np.repeat(rgb, UPSCALE, 0), UPSCALE, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)).to(DEV)
    outs = []
    for k in range(4):   # 4-way rotation test-time augmentation
        o = model(torch.rot90(x, k, (2, 3))).float()
        outs.append(torch.rot90(o, -k, (2, 3)))
    o = torch.stack(outs).mean(0)[0]
    pm = o[:4].softmax(0)
    p = {"roof": (pm[BUILDING] + pm[BOUNDARY] + 0.5 * pm[CONTACT]).cpu().numpy(),
         "contact": pm[CONTACT].cpu().numpy(),
         "int": torch.sigmoid(o[INT]).cpu().numpy(), "dist": torch.sigmoid(o[DIST]).cpu().numpy()}
    lab, sc = unet_decode(p["roof"], p["contact"], p["int"], p["dist"])
    return p, downscale_labels(lab), sc


def run_tests(model, tag):
    model.eval()
    results, store = {}, {}
    for name, samples in (("gandhinagar", load_gandhinagar("test")), ("ramp", load_ramp("test"))):
        rows, recs, maps, pics = [], {}, {}, []
        for sid, rgb, inst, partial in samples:
            inst = inst.astype(np.int32)
            p, lab, sc = predict(model, rgb)
            rows.append(score_tile(lab, inst, partial))
            recs[sid] = pack_instances([lab == v for v in range(1, lab.max() + 1)], sc)
            down = lambda a: (a[::UPSCALE, ::UPSCALE] * 255).astype(np.uint8)
            maps[sid] = np.stack([down(p["roof"]), down(p["contact"]), down(p["int"]), down(p["dist"])])
            if len(pics) < 4:
                pics.append((rgb, [None, inst, lab]))
        results[name] = summarise(rows)
        store[name] = (recs, maps)
        panel(pics, ["image", "label", "U-Net contact"], WORK / f"unet_{name}_{tag}.jpg")
        log(tag, name, results[name])
    return results, store


def main():
    t_start = time.time()
    tr = load_ramp("train")
    gn = crops(load_gandhinagar("train"), 256)
    # Gandhinagar is the target domain but small: repeat it so it is ~20% of each epoch
    rep = max(1, int(0.25 * len(tr) / max(1, len(gn))))
    random.shuffle(tr)
    val = tr[:300]; tr = tr[300:]
    train = tr + gn * rep
    log(f"train: {len(tr)} RAMP chips + {len(gn)} Gandhinagar crops x{rep} = {len(train)}")

    model = build_model()
    dl = DataLoader(Chips(train, True), batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True,
                    persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EPOCHS * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    hist = []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = n = 0
        for x, mm, it, ds, wm in dl:
            x, mm, it, ds, wm = (a.to(DEV, non_blocking=True) for a in (x, mm, it, ds, wm))
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = loss_fn(out.float(), mm, it, ds, wm)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); tot += loss.item(); n += 1
        model.eval()
        rows = [score_tile(predict(model, rgb)[1], inst.astype(np.int32)) for _, rgb, inst, _ in val[:150]]
        v = summarise(rows)
        hist.append({"epoch": ep + 1, "loss": round(tot / n, 4), "min": round((time.time() - t0) / 60, 1), **v})
        log(hist[-1])
        torch.save({"state_dict": model.state_dict(), "n_out": N_OUT, "mean": MEAN.tolist(), "std": STD.tolist(),
                    "upscale": UPSCALE, "arch": "unet_resnet34_multimask"}, WORK / "unet_contact.pt")
        if (time.time() - t_start) / 3600 > BUDGET_H:
            log("time budget reached, stopping"); break

    results, store = run_tests(model, "final")
    np.save(WORK / "pred_unet.npy", {k: v[0] for k, v in store.items()}, allow_pickle=True)
    np.save(WORK / "maps_unet.npy", {k: v[1] for k, v in store.items()}, allow_pickle=True)
    (WORK / "unet_results.json").write_text(json.dumps({"history": hist, "test": results}, indent=2))
    log("done in", round((time.time() - t_start) / 3600, 2), "h")


if __name__ == "__main__":
    main()
