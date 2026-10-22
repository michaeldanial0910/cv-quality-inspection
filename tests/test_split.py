"""Leakage guards on the committed split manifest (no images needed)."""
import hashlib

import pandas as pd
import pytest

from src.config import MANIFEST, SPLIT_FRACTIONS

m = pd.read_csv(MANIFEST) if MANIFEST.exists() else None
needs = pytest.mark.skipif(m is None, reason="run src.split first")


@needs
def test_each_duplicate_group_in_one_split():
    assert m.groupby("dup_group")["split"].nunique().max() == 1


@needs
def test_no_identical_file_in_two_splits():
    assert m.groupby("file_sha256")["split"].nunique().max() == 1


@needs
def test_every_image_assigned_once():
    assert m["path"].is_unique and set(m["split"]) == {"train", "val", "test"} and len(m) == 1300


@needs
def test_split_sizes_and_defect_rate():
    for s, frac in SPLIT_FRACTIONS.items():
        part = m[m["split"] == s]
        assert abs(len(part) / len(m) - frac) < 0.02
        assert abs(part["label"].mean() - m["label"].mean()) < 0.02


@needs
def test_colab_notebook_embeds_this_manifest():
    from pathlib import Path
    nb = Path(MANIFEST).parents[1] / "notebooks" / "02_train_colab.ipynb"
    if not nb.exists():
        pytest.skip("notebook not generated")
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() in nb.read_text()
