"""Train the dense-settlement roof model: semantic classes + roof edge + interior seed.

Why three outputs instead of one class map: in a dense Indian block two touching
houses are both "building", so a single class map fuses them into one shape. Here
the network also predicts the roof edge (including the seam between neighbours)
and a shrunk roof interior, so inference can seed one instance per house and grow
it back to the predicted edge. That is what gives separated roofs with outlines on
the wall instead of one smoothed blob.

Input tiles come from training/dense_instance/prepare (SAM-refined open footprints,
checked by eye before training). Each .npz holds:
    rgb (H,W,3) uint8, inst int16, building/interior/edge/veg/road uint8, dist float32

Semantic supervision is deliberately partial: building, road and vegetation pixels
are supervised, everything else is ignored, so fine-tuning cannot wipe out the
tree / low-vegetation / clutter knowledge the Potsdam model already has.

    python train.py --tiles /kaggle/input/cadastraai-dense/train_dense \
                    --init /kaggle/input/cadastraai-models/unet_potsdam.pt \
                    --epochs 24 --batch 8
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

CLASSES = ["clutter", "building", "road_impervious", "low_vegetation", "tree"]
N_SEM = len(CLASSES)
IGNORE = -100
NOT_BUILDING = -2   # known "not a roof", class unknown (complete-label tiles only)
# channel layout of the head: 5 semantic logits, then edge, interior, distance
SEM, EDGE, INT, DIST = slice(0, N_SEM), N_SEM, N_SEM + 1, N_SEM + 2
N_OUT = N_SEM + 3
PATCH = 512


class Tiles(Dataset):
    """Random 512 px crops with flips and 90 degree rotations."""

    def __init__(self, files, train=True, crops_per_tile=4):
        self.files = files
        self.train = train
        self.crops = crops_per_tile if train else 1

    def __len__(self):
        return len(self.files) * self.crops

    def __getitem__(self, i):
        d = np.load(self.files[i // self.crops])
        rgb = d["rgb"]
        h, w = rgb.shape[:2]
        if self.train:
            y = np.random.randint(0, max(1, h - PATCH))
            x = np.random.randint(0, max(1, w - PATCH))
        else:
            y = max(0, (h - PATCH) // 2)
            x = max(0, (w - PATCH) // 2)
        sl = (slice(y, y + PATCH), slice(x, x + PATCH))

        rgb = rgb[sl].astype(np.float32) / 255.0
        building = d["building"][sl].astype(np.int64)
        road = d["road"][sl].astype(bool)
        veg = d["veg"][sl].astype(bool)
        edge = d["edge"][sl].astype(np.float32)
        interior = d["interior"][sl].astype(np.float32)
        dist = d["dist"][sl].astype(np.float32)
        inst = d["inst"][sl].astype(np.int32)

        if "complete" in d.files:
            # hand-digitised tiles with every building outlined (WHU): off the
            # roofs we KNOW it is not a building, even if we don't know whether it
            # is road or clutter. NOT_BUILDING marks that for the not-a-roof loss.
            sem = np.full(building.shape, NOT_BUILDING, np.int64)
            sem[veg] = CLASSES.index("low_vegetation")
            sem[building > 0] = CLASSES.index("building")
        elif "negative" in d.files:
            # hard negative: a tile we checked by eye and know holds no buildings
            # (ploughed fields, barren scrub). Here "not a roof" is supervised, which
            # is the whole point — these are the tiles that stop false detections.
            sem = np.full(building.shape, CLASSES.index("clutter"), np.int64)
            sem[veg] = CLASSES.index("low_vegetation")
        else:
            sem = np.full(building.shape, IGNORE, np.int64)
            sem[veg] = CLASSES.index("low_vegetation")
            sem[road] = CLASSES.index("road_impervious")
            sem[building > 0] = CLASSES.index("building")

        if "partial" in d.files:
            # mostly-labelled tiles (Gandhinagar): an unlabelled house is not evidence of
            # "no roof", so the edge/seed/distance outputs only learn near labelled
            # houses and on vegetation. -1 marks "unknown" for the masked losses.
            known = ndi.binary_dilation(building > 0, iterations=10) | veg
            edge = np.where(known, edge, -1).astype(np.float32)
            interior = np.where(known, interior, -1).astype(np.float32)
            dist = np.where(known, dist, -1).astype(np.float32)

        if self.train:
            k = np.random.randint(4)
            if k:
                rgb, sem, edge, interior, dist, inst = (
                    np.rot90(a, k, (0, 1)).copy() for a in (rgb, sem, edge, interior, dist, inst))
            if np.random.rand() < 0.5:
                rgb, sem, edge, interior, dist, inst = (
                    np.flip(a, 1).copy() for a in (rgb, sem, edge, interior, dist, inst))
            # mild photometric jitter: Indian roofs vary a lot in brightness
            rgb = np.clip(rgb * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)

        # the deployed model takes RGB + height; these drone tiles have no DSM, so
        # the height channel is zero here, exactly as segment.py feeds it for
        # colour-only surveys.
        x4 = np.dstack([rgb, np.zeros(rgb.shape[:2], np.float32)])
        x4 = ((x4 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)
        return (torch.from_numpy(x4),
                torch.from_numpy(sem),
                torch.from_numpy(edge)[None],
                torch.from_numpy(interior)[None],
                torch.from_numpy(dist)[None],
                torch.from_numpy(inst.astype(np.int32)))


# normalisation must match backend/segment.py exactly, or inference sees different
# inputs than training did; both come from the warm-start checkpoint.
MEAN = np.array([0.485, 0.456, 0.406, 0.0], np.float32)
STD = np.array([0.229, 0.224, 0.225, 1.0], np.float32)


def build_model(init_ckpt=None):
    global MEAN, STD
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=4, classes=N_OUT)
    if init_ckpt:
        ck = torch.load(init_ckpt, map_location="cpu")
        sd = ck.get("state_dict") or ck.get("model") or ck
        if "mean" in ck and "std" in ck:
            MEAN = np.array(ck["mean"], np.float32)
            STD = np.array(ck["std"], np.float32)
        own = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.load_state_dict(own)
        print(f"[init] reused {loaded}/{len(own)} tensors from {Path(init_ckpt).name} "
              f"(the final layer grows from {N_SEM} to {N_OUT} channels, so it starts fresh)")
        # the head is 3 tensors; anything less means the key names did not match
        if loaded < len(own) - 4:
            raise SystemExit(f"warm start matched only {loaded} tensors — check the checkpoint format")
    return model


def masked_bce(logit, target, weight=None):
    m = (target >= 0).float()
    if weight is not None:
        m = m * weight
    l = F.binary_cross_entropy_with_logits(logit, target.clamp_min(0), reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1)


def seam_weight(inst, boost=4.0):
    """Pixels where two different houses meet get extra loss weight: the shared wall
    between neighbours is the boundary the model most needs to learn."""
    inst = inst.float()[:, None]
    big = 1e6
    a = torch.where(inst > 0, inst, torch.full_like(inst, big))
    mx = F.max_pool2d(inst, 5, 1, 2)
    mn = -F.max_pool2d(-a, 5, 1, 2)
    seam = (inst > 0) & (mn < big) & (mx != mn)
    return 1.0 + boost * seam.float()


def masked_dice(logit, target, eps=1.0):
    m = (target >= 0).float()
    p = torch.sigmoid(logit) * m
    t = target.clamp_min(0) * m
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def dice_loss(logit, target, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum() + eps
    den = p.sum() + target.sum() + eps
    return 1 - num / den


def instances_from_prediction(prob_sem, prob_edge, prob_int, pred_dist, min_px=120):
    """Seed one instance per predicted interior blob, grow to the predicted edge."""
    building = prob_sem.argmax(0) == CLASSES.index("building")
    seeds = (prob_int > 0.5) & building
    seeds = ndi.binary_opening(seeds, np.ones((3, 3), bool))
    markers, n = ndi.label(seeds)
    if n == 0:
        return np.zeros_like(markers), 0
    # cost surface: high on predicted edges, low inside roofs
    cost = prob_edge - 0.5 * pred_dist
    labels = watershed(cost, markers=markers, mask=building)
    for v in range(1, labels.max() + 1):
        m = labels == v
        if m.sum() < min_px:
            labels[m] = 0
    return labels, int(len(np.unique(labels)) - 1)


def score(model, loader, device):
    """Building IoU, footprint recall, and the merge rate that exposes dense-area failure."""
    model.eval()
    inter = union = 0
    found = total = 0
    merged = pairs = 0
    edge_tp = edge_fp = edge_fn = 0
    with torch.no_grad():
        for x, sem, edge, interior, dist, inst in loader:
            out = model(x.to(device)).float().cpu()
            for b in range(out.shape[0]):
                ps = out[b, SEM].softmax(0).numpy()
                pe = torch.sigmoid(out[b, EDGE]).numpy()
                pi = torch.sigmoid(out[b, INT]).numpy()
                pd = out[b, DIST].clamp(0, 1).numpy()
                pred_b = ps.argmax(0) == CLASSES.index("building")
                true_b = (sem[b].numpy() == CLASSES.index("building"))
                inter += (pred_b & true_b).sum()
                union += (pred_b | true_b).sum()

                e_true = edge[b, 0].numpy() > 0.5
                e_pred = pe > 0.5
                edge_tp += (e_pred & e_true).sum()
                edge_fp += (e_pred & ~e_true).sum()
                edge_fn += (~e_pred & e_true).sum()

                labels, _ = instances_from_prediction(ps, pe, pi, pd)
                gt = inst[b].numpy()
                for v in np.unique(gt):
                    if v == 0:
                        continue
                    m = gt == v
                    total += 1
                    hit = labels[m]
                    hit = hit[hit > 0]
                    if hit.size and (hit == np.bincount(hit).argmax()).sum() >= 0.5 * m.sum():
                        found += 1
                # merge rate: do two neighbouring ground-truth roofs share one predicted id?
                ids = [v for v in np.unique(gt) if v]
                for v in ids:
                    m = gt == v
                    ring = ndi.binary_dilation(m, np.ones((9, 9), bool)) & ~m
                    for w in np.unique(gt[ring]):
                        if w == 0 or w <= v:
                            continue
                        pairs += 1
                        a = labels[m]
                        c = labels[gt == w]
                        a = a[a > 0]
                        c = c[c > 0]
                        if a.size and c.size and np.bincount(a).argmax() == np.bincount(c).argmax():
                            merged += 1
    f1 = 2 * edge_tp / max(1, 2 * edge_tp + edge_fp + edge_fn)
    return {
        "building_iou": float(inter / max(1, union)),
        "footprint_recall": float(found / max(1, total)),
        "footprints": int(total),
        "merge_rate": float(merged / max(1, pairs)),
        "neighbour_pairs": int(pairs),
        "edge_f1": float(f1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from (unet_potsdam.pt)")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--out", default="unet_dense_instance.pt")
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--gsd", type=float, default=0.05, help="ground sample distance of the tiles, written into the checkpoint")
    ap.add_argument("--val-tiles", default=None, help="a separate validation folder (e.g. WHU's own val split)")
    ap.add_argument("--val-max", type=int, default=0, help="score only this many val tiles per epoch (all at the end)")
    ap.add_argument("--crops", type=int, default=4, help="random crops per tile per epoch (1 when tiles are already 512 px)")
    ap.add_argument("--seam-boost", type=float, default=0.0, help="extra edge-loss weight where two houses touch (0 = off)")
    args = ap.parse_args()

    files = sorted(Path(args.tiles).glob("*.npz"))
    if args.max_tiles:
        files = files[: args.max_tiles]
    if not files:
        raise SystemExit(f"no .npz tiles in {args.tiles}")
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    if args.val_tiles:
        train_files, val_files = files, sorted(Path(args.val_tiles).glob("*.npz"))
    else:
        n_val = max(1, int(len(files) * args.holdout))
        val_files, train_files = files[:n_val], files[n_val:]
    all_val = val_files
    if args.val_max:
        val_files = val_files[: args.val_max]
    print(f"tiles: {len(train_files)} train, {len(all_val)} held out ({len(val_files)} scored per epoch)")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(args.init).to(device)
    tl = DataLoader(Tiles(train_files, True, crops_per_tile=args.crops), batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(Tiles(val_files, False), batch_size=max(1, args.batch // 2), num_workers=2)

    # buildings dominate these tiles; weight the rarer classes up a little
    w = torch.tensor([1.0, 1.0, 1.5, 1.2, 1.2], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(tl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps) if steps > 6 else None
    history = []
    best = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = {"sem": 0.0, "edge": 0.0, "int": 0.0, "dist": 0.0, "n": 0}
        for x, sem, edge, interior, dist, inst_b in tl:
            x, sem = x.to(device), sem.to(device)
            edge, interior, dist = edge.to(device), interior.to(device), dist.to(device)
            out = model(x)
            notb = sem == NOT_BUILDING
            l_sem = F.cross_entropy(out[:, SEM], sem.masked_fill(notb, IGNORE), weight=w, ignore_index=IGNORE)
            if notb.any():
                # push the building probability down where we know there is no roof
                p_b = out[:, SEM].softmax(1)[:, CLASSES.index("building")]
                l_sem = l_sem + (-torch.log((1 - p_b[notb]).clamp_min(1e-6))).mean()
            # targets of -1 are "unknown" (partly-labelled tiles) and carry no loss
            sw = seam_weight(inst_b.to(device), args.seam_boost) if args.seam_boost else None
            l_edge = masked_bce(out[:, EDGE:EDGE + 1], edge, sw) + masked_dice(out[:, EDGE:EDGE + 1], edge)
            l_int = masked_bce(out[:, INT:INT + 1], interior) + masked_dice(out[:, INT:INT + 1], interior)
            md = (dist >= 0).float()
            l_dist = ((out[:, DIST:DIST + 1].clamp(0, 1) - dist.clamp_min(0)).abs() * md).sum() / md.sum().clamp_min(1)
            # edges are thin and are the whole point of this model, so they carry the most weight
            loss = l_sem + 2.0 * l_edge + 1.0 * l_int + 0.5 * l_dist
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            run["sem"] += l_sem.item(); run["edge"] += l_edge.item()
            run["int"] += l_int.item(); run["dist"] += l_dist.item(); run["n"] += 1

        m = score(model, vl, device)
        row = {"epoch": ep, "seconds": round(time.time() - t0, 1),
               **{k: round(v / max(1, run["n"]), 4) for k, v in run.items() if k != "n"}, **m}
        history.append(row)
        print(f"ep{ep:02d} {row['seconds']:5.0f}s  sem {row['sem']:.3f} edge {row['edge']:.3f} "
              f"int {row['int']:.3f} dist {row['dist']:.3f} | IoU {m['building_iou']:.3f} "
              f"footprints {m['footprint_recall']:.2f} merge {m['merge_rate']:.2f} edgeF1 {m['edge_f1']:.2f}")

        # The held-out set is small, so this score jumps around between epochs;
        # keep the pick as a separate *_best.pt, and ship the final epoch.
        key = m["footprint_recall"] - m["merge_rate"]
        if key > best:
            best = key
            save_ckpt(model, args, ep, m, Path(args.out).with_name(Path(args.out).stem + "_best.pt"))
            print("      saved best-by-score checkpoint")

    save_ckpt(model, args, args.epochs, history[-1], Path(args.out))
    print(f"saved final epoch to {args.out}")
    if len(all_val) > len(val_files):
        full = score(model, DataLoader(Tiles(all_val, False), batch_size=max(1, args.batch // 2), num_workers=2), device)
        history.append({"epoch": "final_full_val", **full})
        print("final model on all", len(all_val), "held-out tiles:", full)
    Path(args.out).with_name(Path(args.out).stem + "_metrics.json").write_text(
        json.dumps({"history": history, "best": max(history, key=lambda r: r["footprint_recall"] - r["merge_rate"])}, indent=2))
    print("done. best:", max(history, key=lambda r: r["footprint_recall"] - r["merge_rate"]))
    save_panel(model, val_files, device, Path(args.out).with_name(Path(args.out).stem + "_samples.png"))


def save_ckpt(model, args, epoch, metrics, path):
    # same fields backend/segment.py reads, plus the extra head layout
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "in_channels": 4, "arch": "unet_resnet34", "gsd_m": args.gsd,
                "mean": MEAN.tolist(), "std": STD.tolist(),
                "n_out": N_OUT, "heads": {"semantic": [0, N_SEM], "edge": EDGE,
                                          "interior": INT, "distance": DIST},
                "epoch": epoch, "metrics": metrics}, path)


def save_panel(model, files, device, out_png, n=4):
    """Always look at the masks: metrics from weak labels can flatter a bad model.
    Columns: image, label instances, predicted instances, predicted roof edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    ds = Tiles(files[:n], train=False)
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(len(ds), 4, figsize=(17, 4.3 * len(ds)))
    axes = np.atleast_2d(axes)
    for i in range(len(ds)):
        x, sem, edge, interior, dist, inst = ds[i]
        rgb = np.load(files[i])["rgb"]
        h = rgb.shape[0]
        y0 = max(0, (h - PATCH) // 2)
        rgb = rgb[y0:y0 + PATCH, y0:y0 + PATCH]
        with torch.no_grad():
            out = model(x[None].to(device)).float().cpu()[0]
        ps = out[SEM].softmax(0).numpy()
        pe = torch.sigmoid(out[EDGE]).numpy()
        pi = torch.sigmoid(out[INT]).numpy()
        pd = out[DIST].clamp(0, 1).numpy()
        labels, n_inst = instances_from_prediction(ps, pe, pi, pd)
        gt = inst.numpy()
        for c, (img, title) in enumerate([
                (None, Path(files[i]).stem),
                (gt, f"label: {int(gt.max())} roofs"),
                (labels, f"predicted: {n_inst} roofs"),
                (pe, "predicted roof edge")]):
            ax = axes[i, c]
            ax.imshow(rgb)
            if img is not None and c < 3:
                cols = (rng.random((int(img.max()) + 2, 3)) * 255).astype(np.uint8)
                cols[0] = 0
                lay = cols[np.clip(img, 0, None)]
                m = img > 0
                ov = rgb.copy()
                ov[m] = (0.5 * ov[m] + 0.5 * lay[m]).astype(np.uint8)
                ax.imshow(ov)
            elif img is not None:
                ax.imshow(img, cmap="inferno", alpha=0.65)
            ax.set_title(title, color="w", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.patch.set_facecolor("#11150F")
    fig.tight_layout()
    fig.savefig(out_png, dpi=85, facecolor=fig.get_facecolor())
    print("wrote", out_png)


if __name__ == "__main__":
    main()
