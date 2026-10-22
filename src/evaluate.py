"""Compare models A/B/C on the same test images and fix the decision policy on validation.

Inputs: reports/predictions/{hog_lr, linear_probe, finetuned}.csv (val + test scores).
Every threshold and the temperature are chosen on validation rows; test rows are only scored.

    python -m src.evaluate
"""
import json

import numpy as np
import pandas as pd

from src.config import (AUTO_REJECT_PRECISION, MODELS, REPORTS, REVIEW_BAND_RECALL, TARGET_DEFECT_RECALL)
from src.metrics import (at_threshold, bootstrap, ece, fit_temperature, paired_bootstrap, ranking,
                         threshold_for_precision, threshold_for_recall)
from sklearn.metrics import average_precision_score

MODEL_NAMES = {"hog_lr": "A · HOG + logistic regression (no transfer)",
               "linear_probe": "B · frozen EfficientNet-B0 + logistic head",
               "finetuned": "C · fine-tuned EfficientNet-B0"}
PRED = REPORTS / "predictions"


def load() -> dict:
    out = {}
    for k in MODEL_NAMES:
        p = PRED / f"{k}.csv"
        if p.exists():
            out[k] = pd.read_csv(p)
    return out


def recall_at(t):
    return lambda y, s: at_threshold(y, s, t)["defect_recall"]


def frr_at(t):
    return lambda y, s: at_threshold(y, s, t)["false_reject_rate"]


def main():
    preds = load()
    if not preds:
        raise SystemExit("No predictions yet: run src.baseline and the Colab notebook first.")
    test_paths = None
    rows, summary = [], {"models": {}}
    for k, df in preds.items():
        val, test = df[df["split"] == "val"], df[df["split"] == "test"].sort_values("path")
        if test_paths is None:
            test_paths = test["path"].tolist()
        assert test["path"].tolist() == test_paths, f"{k} was scored on different test images"
        thr = threshold_for_recall(val["label"], val["score"], TARGET_DEFECT_RECALL)
        r = {"model": k, "name": MODEL_NAMES[k], "threshold": thr, **ranking(test["label"], test["score"]),
             **at_threshold(test["label"], test["score"], thr)}
        r["pr_auc_ci"] = bootstrap(average_precision_score, test["label"].values, test["score"].values)[1:]
        r["recall_ci"] = bootstrap(recall_at(thr), test["label"].values, test["score"].values)[1:]
        r["frr_ci"] = bootstrap(frr_at(thr), test["label"].values, test["score"].values)[1:]
        r["acc_at_0.5"] = at_threshold(test["label"], test["score"], 0.5)["accuracy"]
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "metrics.csv", index=False)
    summary["models"] = {r["model"]: {k: r[k] for k in ["threshold", "pr_auc", "pr_auc_ci", "roc_auc", "defect_recall",
                                                        "recall_ci", "missed_defects", "false_reject_rate", "frr_ci",
                                                        "precision", "acc_at_0.5"]} for r in rows}

    # paired comparisons on identical test images
    t = {k: df[df["split"] == "test"].sort_values("path") for k, df in preds.items()}
    y = next(iter(t.values()))["label"].values
    pairs = [(a, b) for a, b in [("finetuned", "linear_probe"), ("finetuned", "hog_lr"), ("linear_probe", "hog_lr")]
             if a in t and b in t]
    summary["paired_pr_auc"] = {f"{a} - {b}": paired_bootstrap(average_precision_score, y, t[a]["score"].values,
                                                               t[b]["score"].values) for a, b in pairs}

    # decision policy for the served model (C if present): temperature + three zones, all on validation
    served = "finetuned" if "finetuned" in preds else next(iter(preds))
    df = preds[served]
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    temp = fit_temperature(val["logit"], val["label"]) if "logit" in df else 1.0
    cal = (lambda d: 1 / (1 + np.exp(-d["logit"] / temp))) if "logit" in df else (lambda d: d["score"])
    pv, pt = cal(val), cal(test)
    policy = {"model": served, "temperature": temp,
              "review_from": threshold_for_recall(val["label"], pv, REVIEW_BAND_RECALL),
              "reject_from": max(threshold_for_precision(val["label"], pv, AUTO_REJECT_PRECISION),
                                 threshold_for_recall(val["label"], pv, TARGET_DEFECT_RECALL))}
    old = json.loads((MODELS / "policy.json").read_text()) if (MODELS / "policy.json").exists() else {}
    if "ood_distance" in old:
        policy["ood_distance"] = old["ood_distance"]
    (MODELS / "policy.json").write_text(json.dumps(policy, indent=2))
    zone = np.where(pt < policy["review_from"], "auto-pass", np.where(pt >= policy["reject_from"], "auto-reject", "human review"))
    z = pd.DataFrame({"zone": zone, "label": test["label"].values})
    summary["policy"] = policy
    summary["zones_test"] = {zn: {"n": int((z.zone == zn).sum()), "defective": int(z[z.zone == zn].label.sum()),
                                  "ok": int((z[z.zone == zn].label == 0).sum())}
                             for zn in ["auto-pass", "human review", "auto-reject"]}
    summary["calibration"] = {"ece_val_before": ece(val["label"], val["score"]), "ece_val_after": ece(val["label"], pv),
                              "ece_test_after": ece(test["label"], pt)}
    df.assign(prob=cal(df)).to_csv(PRED / f"{served}_calibrated.csv", index=False)
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(table[["model", "pr_auc", "roc_auc", "threshold", "defect_recall", "missed_defects", "false_reject_rate",
                 "acc_at_0.5"]].round(4).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ["paired_pr_auc", "policy", "zones_test", "calibration"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
