"""Find near-duplicate photos (the same physical part shot twice, or a rotated/mirrored copy).

Why this is needed: the 1,300 "originals" include several photos of the same casting, rotated
by arbitrary angles. A random split would put one photo in train and another in test, and the
model would be graded on a part it has already seen.

How: each part is centred on its edge energy and unwrapped into polar coordinates (radius x
angle) around that centre. A rotation of the part is then a circular shift along the angle
axis, so the best normalised correlation over all shifts (computed with an FFT), and over the
mirror image, is a rotation- and flip-invariant similarity in [-1, 1]. The per-radius mean is
removed first, so the shared shape of every impeller does not count as similarity.

Threshold (DUP_SIM = 0.97) was set by looking at image pairs across the similarity range
(reports/figures/duplicate_pairs.png): at >= 0.97 pairs are the same part (no group mixes the two
labels); at 0.95 different parts under the same lighting start to merge.

    python -m src.dedupe
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from src.config import DUP_SIM, EXTRACTED, FIGURES, PROCESSED, REPORTS
from src.data import load_gray

RADII, ANGLES = 40, 180


def polar(path) -> np.ndarray:
    x = load_gray(EXTRACTED / path, 128)
    g = ndimage.gaussian_filter(x, 1.0)
    e = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    e = e * (e > np.quantile(e, 0.8))                      # strongest edges = the part, not the background
    yy, xx = np.indices(e.shape)
    cy, cx = (e * yy).sum() / e.sum(), (e * xx).sum() / e.sum()
    r = np.linspace(4, 60, RADII)[:, None]
    t = np.linspace(0, 2 * np.pi, ANGLES, endpoint=False)[None, :]
    return ndimage.map_coordinates(x, [cy + r * np.sin(t), cx + r * np.cos(t)], order=1, mode="nearest")


def descriptors(paths) -> np.ndarray:
    p = np.stack([polar(x) for x in paths]).astype(np.float32)
    p -= p.mean(axis=2, keepdims=True)
    p /= np.linalg.norm(p.reshape(len(p), -1), axis=1)[:, None, None] + 1e-6
    return p


def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) best correlation over all rotations and the mirror image."""
    fb = np.conj(np.fft.rfft(b, axis=2))
    fa, fam = np.fft.rfft(a, axis=2), np.fft.rfft(a[:, :, ::-1], axis=2)
    out = np.empty((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        c = np.fft.irfft((fa[i][None] * fb).sum(1), n=ANGLES, axis=1).max(1)
        cm = np.fft.irfft((fam[i][None] * fb).sum(1), n=ANGLES, axis=1).max(1)
        out[i] = np.maximum(c, cm)
    return out


def groups(n: int, pairs) -> np.ndarray:
    """Connected components (union-find): images linked by a near-duplicate pair share a group."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        parent[find(i)] = find(j)
    _, g = np.unique([find(i) for i in range(n)], return_inverse=True)
    return g


def pair_sheet(paths, labels, sim, fn, bands=((0.97, 1.0), (0.95, 0.97), (0.92, 0.95), (0.88, 0.92), (0.8, 0.88))):
    """Two random pairs per similarity band, side by side, for choosing/justifying the threshold."""
    iu = np.triu_indices(len(sim), 1)
    v = sim[iu]
    rng = np.random.default_rng(0)
    rows = []
    for lo, hi in bands:
        idx = np.nonzero((v >= lo) & (v < hi))[0]
        rows += [(iu[0][k], iu[1][k], v[k]) for k in rng.choice(idx, size=min(2, len(idx)), replace=False)]
    s = 180
    sheet = Image.new("RGB", (2 * s + 230, s * len(rows)), "white")
    d = ImageDraw.Draw(sheet)
    name = {0: "ok", 1: "defective"}
    for r, (i, j, val) in enumerate(rows):
        for c, k in enumerate((i, j)):
            sheet.paste(Image.open(EXTRACTED / paths[k]).convert("RGB").resize((s, s)), (c * (s + 4), r * s))
        d.text((2 * s + 16, r * s + s // 2 - 14), f"similarity {val:.3f}", fill=(28, 35, 51))
        d.text((2 * s + 16, r * s + s // 2 + 4), f"{name[labels[i]]} / {name[labels[j]]}", fill=(90, 100, 120))
        d.text((2 * s + 16, r * s + s // 2 + 22), "same part" if val >= DUP_SIM else "kept apart", fill=(90, 100, 120))
    sheet.save(fn)


def hist(v, edges=(-1, 0.5, 0.7, 0.8, 0.88, 0.92, 0.95, 0.97, 0.99, 1.01)) -> dict:
    c, _ = np.histogram(v, bins=list(edges))
    return {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(x) for i, x in enumerate(c)}


def main():
    inv = pd.read_csv(PROCESSED / "inventory.csv", dtype={"source": str})
    orig = inv[inv["source"] == "512"].reset_index(drop=True)
    k300 = inv[inv["source"] == "300"].reset_index(drop=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. originals: group photos of the same part
    d_orig = descriptors(orig["path"])
    s = similarity(d_orig, d_orig)
    np.fill_diagonal(s, -1)
    iu = np.triu_indices(len(s), 1)
    keep = s[iu] >= DUP_SIM
    pairs = list(zip(iu[0][keep], iu[1][keep]))
    orig["dup_group"] = groups(len(orig), pairs)
    orig["nn_sim"] = s.max(1)
    pd.DataFrame({"path_a": orig["path"].values[iu[0][keep]], "path_b": orig["path"].values[iu[1][keep]],
                  "similarity": s[iu][keep].round(4), "label_a": orig["label"].values[iu[0][keep]],
                  "label_b": orig["label"].values[iu[1][keep]]}).to_csv(REPORTS / "duplicates_originals.csv", index=False)
    orig[["path", "label", "file_sha256", "dup_group", "nn_sim"]].to_csv(PROCESSED / "originals_groups.csv", index=False)
    pair_sheet(orig["path"].values, orig["label"].values, s, FIGURES / "duplicate_pairs.png")
    sizes = orig.groupby("dup_group").size()

    # 2. Kaggle's own 300x300 train/test split: does each test image have a copy in train?
    tr, te = k300[k300["kaggle_split"] == "train"], k300[k300["kaggle_split"] == "test"]
    s_k = similarity(descriptors(te["path"]), descriptors(tr["path"]))
    te = te.assign(nn_train_sim=s_k.max(1), nn_train_path=tr["path"].values[s_k.argmax(1)],
                   exact_file_in_train=te["file_sha256"].isin(set(tr["file_sha256"])))
    te.to_csv(REPORTS / "kaggle_split_leakage.csv", index=False)
    pair_sheet_te = te.sort_values("nn_train_sim", ascending=False).head(1)

    summary = {
        "method": "rotation/flip-invariant polar correlation, threshold %.2f" % DUP_SIM,
        "originals": int(len(orig)),
        "originals_nn_similarity_hist": hist(orig["nn_sim"]),
        "originals_duplicate_pairs": int(len(pairs)),
        "originals_groups": int(len(sizes)),
        "originals_with_a_near_copy": int((sizes[orig["dup_group"]].values > 1).sum()),
        "largest_group": int(sizes.max()),
        "groups_with_both_labels": int(orig.groupby("dup_group")["label"].nunique().gt(1).sum()),
        "kaggle300_test_images": int(len(te)),
        "kaggle300_test_exact_file_copy_in_train": int(te["exact_file_in_train"].sum()),
        "kaggle300_test_with_near_copy_in_train": int((te["nn_train_sim"] >= DUP_SIM).sum()),
        "kaggle300_test_nn_train_similarity_hist": hist(te["nn_train_sim"]),
        "kaggle300_example_pair": pair_sheet_te[["path", "nn_train_path", "nn_train_sim"]].to_dict("records"),
    }
    (REPORTS / "duplicates_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
