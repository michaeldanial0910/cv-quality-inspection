"""Deployment stress test: how much does the served model degrade when the camera or light changes?

Each test image is perturbed (brightness, contrast, blur, noise, small rotation, shift, JPEG
quality) at increasing severity and re-scored. The decision threshold is the one fixed on
validation, so the numbers show what would happen on a line where conditions drift and nobody
recalibrates. Also measures CPU latency and model size.

    python -m src.robustness                       # everything
    python -m src.robustness brightness contrast   # only these (merged into reports/robustness.csv)
"""
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score

from src.config import EXTRACTED, MANIFEST, MODELS, REPORTS, SEED
from src.infer import Inspector
from src.metrics import at_threshold
from src.preprocess import to_gray_image


def jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("L")


def noise(img, sd, rng):
    a = np.asarray(img, np.float32) + rng.normal(0, sd, (img.height, img.width))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def shift(img, frac):
    dx = int(frac * img.width)
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dx), fillcolor=128)


PERTURB = {
    "brightness": [(0.6, lambda im, r: ImageEnhance.Brightness(im).enhance(0.6)),
                   (0.8, lambda im, r: ImageEnhance.Brightness(im).enhance(0.8)),
                   (1.25, lambda im, r: ImageEnhance.Brightness(im).enhance(1.25)),
                   (1.5, lambda im, r: ImageEnhance.Brightness(im).enhance(1.5))],
    "contrast": [(0.6, lambda im, r: ImageEnhance.Contrast(im).enhance(0.6)),
                 (1.5, lambda im, r: ImageEnhance.Contrast(im).enhance(1.5))],
    "blur (px)": [(1, lambda im, r: im.filter(ImageFilter.GaussianBlur(1))),
                  (2, lambda im, r: im.filter(ImageFilter.GaussianBlur(2))),
                  (4, lambda im, r: im.filter(ImageFilter.GaussianBlur(4)))],
    "noise (sd)": [(10, lambda im, r: noise(im, 10, r)), (25, lambda im, r: noise(im, 25, r))],
    "rotation (deg)": [(10, lambda im, r: im.rotate(10, fillcolor=128)), (45, lambda im, r: im.rotate(45, fillcolor=128))],
    "shift (% of width)": [(5, lambda im, r: shift(im, 0.05)), (15, lambda im, r: shift(im, 0.15))],
    "jpeg quality": [(30, lambda im, r: jpeg(im, 30)), (10, lambda im, r: jpeg(im, 10))],
}


def main():
    ins = Inspector.load()
    m = pd.read_csv(MANIFEST)
    test = m[m["split"] == "test"].reset_index(drop=True)
    imgs = [to_gray_image(EXTRACTED / p) for p in test["path"]]
    y = test["label"].values
    thr = ins.policy.review_from          # anything at/above goes to a person or is rejected: the "flagged" line
    rng = np.random.default_rng(SEED)

    def score(images):
        lg, _ = ins.predict_batch(images)
        return ins.prob(lg)

    only = sys.argv[1:]
    rows = []
    clean = score(imgs)
    base = at_threshold(y, clean, thr)
    rows.append({"perturbation": "none", "severity": 0, "pr_auc": average_precision_score(y, clean),
                 "defect_recall": base["defect_recall"], "missed_defects": base["missed_defects"],
                 "false_reject_rate": base["false_reject_rate"]})
    for name, levels in PERTURB.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        for sev, fn in levels:
            p = score([fn(im, rng) for im in imgs])
            r = at_threshold(y, p, thr)
            rows.append({"perturbation": name, "severity": sev, "pr_auc": average_precision_score(y, p),
                         "defect_recall": r["defect_recall"], "missed_defects": r["missed_defects"],
                         "false_reject_rate": r["false_reject_rate"]})
            print(rows[-1])
    out = pd.DataFrame(rows)
    path = REPORTS / "robustness.csv"
    if only and path.exists():                       # merge a partial run into earlier results
        old = pd.read_csv(path)
        old = old[~old["perturbation"].isin(out["perturbation"])]
        out = pd.concat([old, out]).drop_duplicates(["perturbation", "severity"], keep="last")
    out.to_csv(path, index=False)
    if only:
        return

    one = imgs[0]
    for _ in range(3):
        ins.inspect(one)
    t = []
    for im in imgs[:50]:
        t0 = time.perf_counter()
        ins.inspect(im)
        t.append((time.perf_counter() - t0) * 1000)
    perf = {"cpu_ms_per_image_median": float(np.median(t)), "cpu_ms_per_image_p95": float(np.percentile(t, 95)),
            "cpu_threads": os.cpu_count(), "onnx_mb": round(os.path.getsize(MODELS / "model.onnx") / 1e6, 1)}
    (REPORTS / "performance.json").write_text(json.dumps(perf, indent=2))
    print(perf)


if __name__ == "__main__":
    main()
