"""Shared pieces of the house-separation ensemble: data, targets, instance
encoding, scoring and pictures. Every member kernel and the fusion kernel import
this, so all models are trained on the same crops and scored the same way.

Scale: RAMP chips and the Gandhinagar mosaic are both ~0.3 m. A house is ~30-60 px,
so every member works on images upscaled 2x (UPSCALE), which gives the instance
models enough pixels per roof.
"""
import glob
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

UPSCALE = 2
MIN_HOUSE_PX = 20            # at native 0.3 m scale (~1.8 m2): smaller shapes are noise


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return hits


def load_ramp(split):
    """-> list of (name, rgb uint8 HxWx3, inst int32, partial=False)."""
    out = []
    for f in find(f"ramp/*_{split}.npz"):
        d = np.load(f)
        # read each array once: indexing an NpzFile key decompresses the whole array again
        rgb, inst = d["rgb"], d["inst"]
        city = Path(f).stem.rsplit("_", 1)[0]
        # inst stays int16 (views, ~7 GB for all cities); users cast per sample
        out += [(f"{city}_{i:05d}", rgb[i], inst[i], False) for i in range(len(rgb))]
    return out


def load_gandhinagar(split):
    """Geographic split of the stitched Gandhinagar mosaic (gnm_train_* / gnm_test_*).
    ~10-15% of houses are unlabelled, hence partial=True."""
    out = []
    for f in find(f"gnm_{split}_*.npz"):
        d = np.load(f)
        out.append((Path(f).stem, d["rgb"], d["inst"].astype(np.int32), True))
    return out


def crops(samples, size=256, per_big=None, rng=None):
    """Cut large tiles into size x size crops so every image has the same shape.
    Keeps 256 chips as they are."""
    rng = rng or np.random.default_rng(0)
    out = []
    for name, rgb, inst, partial in samples:
        h, w = inst.shape
        if h == size and w == size:
            out.append((name, rgb, inst, partial))
            continue
        ys = list(range(0, h - size + 1, size // 2)); xs = list(range(0, w - size + 1, size // 2))
        for y in ys:
            for x in xs:
                out.append((f"{name}_{y}_{x}", rgb[y:y + size, x:x + size],
                            relabel(inst[y:y + size, x:x + size]), partial))
    return out


def relabel(inst):
    ids = np.unique(inst); ids = ids[ids > 0]
    lut = np.zeros(int(inst.max()) + 1, np.int32); lut[ids] = np.arange(1, len(ids) + 1)
    return lut[inst]


def upscale(rgb, inst=None):
    rgb2 = np.repeat(np.repeat(rgb, UPSCALE, 0), UPSCALE, 1)
    if inst is None:
        return rgb2
    return rgb2, np.repeat(np.repeat(inst, UPSCALE, 0), UPSCALE, 1)


def downscale_labels(lab):
    """Nearest-neighbour back to native scale (take the top-left of each 2x2 block)."""
    return lab[::UPSCALE, ::UPSCALE]


# ------------------------------------------------------------------ RAMP-style targets
# classes of the multimask (as in the RAMP baseline) at the model's working scale
BG, BUILDING, BOUNDARY, CONTACT = 0, 1, 2, 3


def multimask(inst, boundary_px=2, contact_px=3):
    """background / building / boundary / close_contact.

    contact: any pixel whose neighbourhood holds two different houses -- the shared
    wall between row houses and the narrow gap between close ones. This band is a few
    pixels wide on purpose: a 1-px seam is too thin for the network to learn.
    boundary: the rest of each house's outer ring."""
    big = np.iinfo(np.int32).max
    k = 2 * contact_px + 1
    mx = ndi.maximum_filter(inst, k)
    mn = ndi.minimum_filter(np.where(inst > 0, inst, big), k)
    contact = (mn < big) & (mx > 0) & (mn != mx)
    b = inst > 0
    # ring of each house separately: eroding the union alone would miss shared walls
    edge_any = np.zeros_like(b)
    edge_any[1:] |= inst[1:] != inst[:-1]; edge_any[:-1] |= inst[:-1] != inst[1:]
    edge_any[:, 1:] |= inst[:, 1:] != inst[:, :-1]; edge_any[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    near_edge = ndi.binary_dilation(edge_any, iterations=boundary_px - 1) if boundary_px > 1 else edge_any
    ring = b & near_edge
    mm = np.zeros(inst.shape, np.uint8)
    mm[b] = BUILDING
    mm[ring] = BOUNDARY
    mm[contact] = CONTACT
    return mm


def interior(inst, shrink_px=3):
    """Each house shrunk on its own (Nacala's gap trick), so seeds of neighbours never touch."""
    out = np.zeros(inst.shape, bool)
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        out[sl] |= ndi.binary_erosion(inst[sl] == v, iterations=shrink_px)
    return out


def distance(inst, cap_px=12):
    d = np.zeros(inst.shape, np.float32)
    for v, sl in enumerate(ndi.find_objects(inst), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in sl)
        m = inst[sl] == v
        d[sl] = np.where(m, np.minimum(ndi.distance_transform_edt(m) / cap_px, 1.0), d[sl])
    return d


def weight_map(inst, w0=10.0, sigma=4.0):
    """U-Net paper / Nacala border weight: background pixels squeezed between two
    houses get up to 1+w0 loss weight."""
    ids = [v for v in np.unique(inst) if v]
    w = np.ones(inst.shape, np.float32)
    if len(ids) < 2:
        return w
    bg = inst == 0
    d1 = np.full(inst.shape, np.inf, np.float32); d2 = d1.copy()
    for v in ids:
        dv = ndi.distance_transform_edt(inst != v).astype(np.float32)
        d2 = np.minimum(d2, np.maximum(d1, dv)); d1 = np.minimum(d1, dv)
    w[bg] += w0 * np.exp(-((d1[bg] + d2[bg]) ** 2) / (2 * sigma ** 2))
    return w


def unet_decode(p_roof, p_contact, p_int, dist, seed_thr=0.5, contact_thr=0.35):
    """Contact-aware watershed (SpaceNet-4 winner style): seeds are shrunk roof
    interiors with the contact band cut out; each seed grows back over the roof,
    stopping where the contact band / distance valley says one house ends."""
    from skimage.segmentation import watershed
    roof = p_roof > 0.5
    seeds = (p_int > seed_thr) & (p_contact < contact_thr) & roof
    seeds = ndi.binary_opening(seeds, np.ones((2, 2), bool))
    markers, n = ndi.label(seeds)
    if n == 0:
        return np.zeros(roof.shape, np.int32), np.zeros(0, np.float32)
    lab = watershed(-dist + p_contact, markers=markers, mask=roof)
    sizes = np.bincount(lab.ravel())
    lab[np.isin(lab, np.nonzero(sizes < MIN_HOUSE_PX * UPSCALE ** 2)[0])] = 0
    lab = relabel(lab)
    score = ndi.mean(p_roof, lab, index=np.arange(1, lab.max() + 1)) if lab.max() else np.zeros(0)
    return lab, np.asarray(score, np.float32)


# ------------------------------------------------------------------ instance storage
def pack_instances(masks, scores):
    """masks: list of HxW bool; stored as bbox + packed bits so thousands fit."""
    recs = []
    for m, s in zip(masks, scores):
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        recs.append((float(s), int(y0), int(x0), int(y1 - y0), int(x1 - x0), np.packbits(m[y0:y1, x0:x1])))
    return recs


def unpack_instances(recs, shape):
    masks, scores = [], []
    for s, y0, x0, h, w, bits in recs:
        m = np.zeros(shape, bool)
        m[y0:y0 + h, x0:x0 + w] = np.unpackbits(bits)[: h * w].reshape(h, w).astype(bool)
        masks.append(m); scores.append(s)
    return masks, np.array(scores, np.float32)


def masks_to_labels(masks, scores):
    """Overlapping instances -> one label raster; higher score wins contested pixels."""
    lab = np.zeros(masks[0].shape if masks else (0, 0), np.int32)
    for i in np.argsort(-np.asarray(scores)) if len(masks) else []:
        free = masks[i] & (lab == 0)
        if free.sum() >= MIN_HOUSE_PX:
            lab[free] = i + 1
    return relabel(lab) if lab.size else lab


# ------------------------------------------------------------------ scoring
def score_tile(pred, gt, partial=False):
    """Houses found (IoU>=0.5), outline IoU of found houses, neighbouring house pairs
    that share one predicted shape, and predicted shapes matching no house. On
    partly-labelled tiles an 'extra' shape may be a real but unlabelled house."""
    g = gt.ravel().astype(np.int64); p = pred.ravel().astype(np.int64)
    G, P = int(g.max()) + 1, int(p.max()) + 1
    cont = np.bincount(g * P + p, minlength=G * P).reshape(G, P)
    inter = cont[1:, 1:]
    union = cont[1:].sum(1)[:, None] + cont[:, 1:].sum(0)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    best = iou.max(1) if iou.size else np.zeros(G - 1)
    found = best >= 0.5
    extra = int((iou.max(0) < 0.5).sum()) if iou.size else P - 1
    if partial and iou.size:
        # a predicted shape lying mostly on unlabelled ground is not counted as extra
        on_label = cont[1:, 1:].sum(0) / np.maximum(cont[:, 1:].sum(0), 1)
        extra = int(((iou.max(0) < 0.5) & (on_label > 0.3)).sum())
    dom = np.where(inter.sum(1) > 0, inter.argmax(1) + 1, 0) if inter.size else np.zeros(G - 1, int)
    BIG = 1 << 30
    gmax = ndi.maximum_filter(gt, 7)
    gmin = ndi.minimum_filter(np.where(gt > 0, gt, BIG), 7)
    ok = (gmin < BIG) & (gmax > 0) & (gmin != gmax)
    pairs = set(zip(gmin[ok].tolist(), gmax[ok].tolist()))
    merged = sum(1 for a, b in pairs if dom[a - 1] and dom[a - 1] == dom[b - 1])
    return dict(total=G - 1, found=int(found.sum()), iou_sum=float(best[found].sum()),
                pairs=len(pairs), merged=merged, extra=extra, shapes=P - 1,
                pix_i=int(((pred > 0) & (gt > 0)).sum()), pix_u=int(((pred > 0) | (gt > 0)).sum()))


def summarise(rows):
    s = {k: sum(r[k] for r in rows) for k in rows[0]}
    prec = s["found"] / max(1, s["found"] + s["extra"])
    rec = s["found"] / max(1, s["total"])
    return dict(houses_found=f"{s['found']}/{s['total']}", recall=round(rec, 3),
                precision=round(prec, 3), f1=round(2 * prec * rec / max(1e-9, prec + rec), 3),
                outline_iou=round(s["iou_sum"] / max(1, s["found"]), 3),
                merged_pairs=f"{s['merged']}/{s['pairs']}", merge_rate=round(s["merged"] / max(1, s["pairs"]), 3),
                extra_shapes=s["extra"], building_iou=round(s["pix_i"] / max(1, s["pix_u"]), 3))


# ------------------------------------------------------------------ pictures
def overlay(rgb, lab, seed=3):
    rng = np.random.default_rng(seed)
    cols = (rng.random((int(lab.max()) + 2, 3)) * 255).astype(np.uint8); cols[0] = 0
    ov = rgb.copy(); m = lab > 0
    ov[m] = (0.45 * ov[m] + 0.55 * cols[lab[m]]).astype(np.uint8)
    e = np.zeros(lab.shape, bool)
    e[1:] |= lab[1:] != lab[:-1]; e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    ov[e & (ndi.binary_dilation(m))] = 255
    return ov


def panel(rows, titles, path, scale=1):
    """rows: list of (rgb, [lab or None, ...]); one row per image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(titles)
    fig, ax = plt.subplots(len(rows), n, figsize=(4.2 * n * scale, 4.3 * len(rows) * scale))
    ax = np.atleast_2d(ax)
    for r, (rgb, labs) in enumerate(rows):
        for c, lab in enumerate(labs):
            img = rgb if lab is None else overlay(rgb, lab)
            t = titles[c] if lab is None else f"{titles[c]}: {len(np.unique(lab)) - 1}"
            ax[r, c].imshow(img); ax[r, c].set_title(t, fontsize=10)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
