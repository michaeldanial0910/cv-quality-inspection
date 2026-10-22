"""Verify and unpack the Kaggle zip, then build an inventory of every image.

    python -m src.data
"""
import hashlib
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASSES, EXTRACTED, PROCESSED, SET_300, SET_512, ZIP, ZIP_SHA256


def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def verify_and_extract():
    if not ZIP.exists():
        raise SystemExit(f"Missing {ZIP}. Download it first (README, 'How to run').")
    got = sha256(ZIP)
    if got != ZIP_SHA256:
        raise SystemExit(f"Checksum mismatch for {ZIP.name}: {got}")
    if not SET_512.exists() or not SET_300.exists():
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(EXTRACTED)
    print(f"zip OK ({got[:12]}...), extracted to {EXTRACTED.relative_to(ZIP.parents[2])}")


def load_gray(path, size: int | None = None) -> np.ndarray:
    """Images are stored as RGB JPEGs but are grayscale; load as float32 in [0, 1]."""
    im = Image.open(path).convert("L")
    if size:
        im = im.resize((size, size), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def inventory() -> pd.DataFrame:
    rows = []
    for source, root, subsets in [("512", SET_512, [""]), ("300", SET_300, ["train", "test"])]:
        for sub in subsets:
            for cls, label in CLASSES.items():
                folder = root / sub / cls if sub else root / cls
                for p in sorted(folder.glob("*.jpeg")):
                    rows.append({"source": source, "kaggle_split": sub or None, "label": label,
                                 "path": str(p.relative_to(EXTRACTED)), "file_sha256": sha256(p)})
    df = pd.DataFrame(rows)
    return df


def main():
    verify_and_extract()
    inv = inventory()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    inv.to_csv(PROCESSED / "inventory.csv", index=False)
    print(inv.groupby(["source", "kaggle_split", "label"], dropna=False).size().to_string())
    exact = inv[inv["source"] == "512"]["file_sha256"].duplicated().sum()
    print(f"exact byte-duplicates inside the 512 set: {exact}")


if __name__ == "__main__":
    main()
