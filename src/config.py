"""Paths and run settings shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ZIP = RAW / "real-life-industrial-dataset-of-casting-product.zip"
EXTRACTED = RAW / "casting"
SET_512 = EXTRACTED / "casting_512x512" / "casting_512x512"     # 1,300 original photos (main dataset)
SET_300 = EXTRACTED / "casting_data" / "casting_data"           # 7,348 pre-augmented 300x300 copies
PROCESSED = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "split_manifest.csv"                 # committed: which file is in which split
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Kaggle: ravirajsinh45/real-life-industrial-dataset-of-casting-product (CC BY-NC-ND 4.0).
ZIP_SHA256 = "a11236976a65d77edd41fa3f8a2803a583157bd7401efc5e7ef5532cf3a00437"

CLASSES = {"ok_front": 0, "def_front": 1}   # defective is the positive class
CLASS_NAME = {0: "acceptable", 1: "defective"}

SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Near-duplicate detection (src/dedupe.py): rotation/flip-invariant polar correlation. Pairs at or
# above this are treated as the same physical part and always land in the same split. Set by
# inspecting pairs across the similarity range (reports/figures/duplicate_pairs.png), never by
# looking at a model metric.
DUP_SIM = 0.97

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Decision policy (chosen on validation only).
TARGET_DEFECT_RECALL = 0.99      # a missed defect ships a bad part; a false reject scraps a good one
REVIEW_BAND_RECALL = 0.995       # lower edge of the human-review zone
AUTO_REJECT_PRECISION = 0.99     # upper edge: only auto-reject where this precise on validation
N_BOOT = 2000
