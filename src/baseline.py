"""Model A, the no-transfer-learning floor: HOG features + logistic regression.

HOG (histograms of edge directions in small cells) is the classic hand-made image feature.
If an ImageNet backbone cannot beat it clearly, transfer learning is not earning its cost.
Regularisation strength C is chosen on validation (average precision); test is not touched.

    python -m src.baseline
"""
import json

import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EXTRACTED, MANIFEST, MODELS, REPORTS, SEED
from src.data import load_gray

HOG_SIZE = 128


def hog_features(paths) -> np.ndarray:
    return np.stack([hog(load_gray(EXTRACTED / p, HOG_SIZE), orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), feature_vector=True) for p in paths]).astype(np.float32)


def main():
    m = pd.read_csv(MANIFEST)
    X = {s: hog_features(m.loc[m["split"] == s, "path"]) for s in ["train", "val", "test"]}
    y = {s: m.loc[m["split"] == s, "label"].values for s in ["train", "val", "test"]}
    log = []
    for c in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, random_state=SEED))
        clf.fit(X["train"], y["train"])
        log.append({"C": c, "val_ap": average_precision_score(y["val"], clf.predict_proba(X["val"])[:, 1])})
    log = pd.DataFrame(log)
    best_c = float(log.loc[log["val_ap"].idxmax(), "C"])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000, random_state=SEED))
    clf.fit(X["train"], y["train"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS / "hog_lr.joblib")
    preds = []
    for s in ["val", "test"]:
        part = m.loc[m["split"] == s, ["path", "label", "split"]].copy()
        part["score"] = clf.predict_proba(X[s])[:, 1]
        preds.append(part)
    out = REPORTS / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_csv(out / "hog_lr.csv", index=False)
    log.to_csv(REPORTS / "tuning_hog_lr.csv", index=False)
    print(log.to_string(index=False))
    print(json.dumps({"best_C": best_c, "n_features": int(X["train"].shape[1])}))


if __name__ == "__main__":
    main()
