"""Group-aware, stratified 70/15/15 split of the 1,300 originals -> data/split_manifest.csv.

Every near-duplicate group (src/dedupe.py) goes to exactly one split, so no part is seen in
training and graded in validation/test. Groups are single-label (checked), so stratifying by
group label keeps the defect rate the same in every split.

    python -m src.split
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src.config import MANIFEST, PROCESSED, REPORTS, SEED, SPLIT_FRACTIONS


def assign(groups: pd.DataFrame, seed: int = SEED) -> dict:
    """groups: one row per dup_group with columns label, n. Returns {group: split}."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in groups.groupby("label"):
        g = g.iloc[rng.permutation(len(g))]
        total, cum = g["n"].sum(), 0
        cut_train = SPLIT_FRACTIONS["train"] * total
        cut_val = (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]) * total
        for grp, n in zip(g.index, g["n"]):
            mid = cum + n / 2                       # place a group by its midpoint: no systematic bias
            out[grp] = "train" if mid <= cut_train else "val" if mid <= cut_val else "test"
            cum += n
    return out


def manifest_sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def main():
    g = pd.read_csv(PROCESSED / "originals_groups.csv")
    assert g.groupby("dup_group")["label"].nunique().max() == 1, "a duplicate group mixes labels"
    per_group = g.groupby("dup_group").agg(label=("label", "first"), n=("path", "size"))
    g["split"] = g["dup_group"].map(assign(per_group))
    m = g[["path", "label", "split", "dup_group", "file_sha256"]].sort_values("path").reset_index(drop=True)
    m.to_csv(MANIFEST, index=False)
    summary = {
        "manifest_sha256": manifest_sha256(m),
        "counts": {s: int((m["split"] == s).sum()) for s in ["train", "val", "test"]},
        "defect_rate": {s: round(float(m.loc[m["split"] == s, "label"].mean()), 4) for s in ["train", "val", "test"]},
        "groups": {s: int(m.loc[m["split"] == s, "dup_group"].nunique()) for s in ["train", "val", "test"]},
    }
    (REPORTS / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
