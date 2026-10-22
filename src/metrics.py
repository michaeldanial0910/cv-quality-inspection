"""Classification metrics with defective = positive class, plus bootstrap intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import N_BOOT, SEED


def at_threshold(y, score, thr) -> dict:
    y, pred = np.asarray(y), np.asarray(score) >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "defect_recall": tp / max(tp + fn, 1),          # share of defective parts caught
            "missed_defects": fn,                           # "escapes": bad parts that would ship
            "false_reject_rate": fp / max(fp + tn, 1),      # share of good parts scrapped
            "precision": tp / max(tp + fp, 1),
            "accuracy": (tp + tn) / len(y),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def ranking(y, score) -> dict:
    return {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score))}


def threshold_for_recall(y, score, target) -> float:
    """Highest threshold whose defect recall is still >= target (fewest false rejects at that recall)."""
    y, score = np.asarray(y), np.asarray(score)
    pos = np.sort(score[y == 1])
    k = int(np.floor((1 - target) * len(pos)))      # how many defects we may miss
    return float(pos[k]) if len(pos) else 0.5


def threshold_for_precision(y, score, target) -> float:
    """Lowest threshold at which precision on the flagged set is >= target (1.0 if none)."""
    y, score = np.asarray(y), np.asarray(score)
    for t in np.unique(score):
        sel = score >= t
        if sel.sum() and y[sel].mean() >= target:
            return float(t)
    return 1.0


def ece(y, prob, bins: int = 10) -> float:
    """Expected calibration error: average |confidence - accuracy| over equal-width probability bins."""
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    return float(sum(abs(prob[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def bootstrap(fn, y, score, n_boot: int = N_BOOT, seed: int = SEED):
    """Point estimate and 95% percentile interval of fn(y, score), resampling images."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], score[i]))
    return float(fn(y, score)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(fn, y, a, b, n_boot: int = N_BOOT, seed: int = SEED):
    """fn(y, a) - fn(y, b) on the same resampled images: is model a better than b?"""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    return float(fn(y, a) - fn(y, b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fit_temperature(logits, y) -> float:
    """Temperature T minimising validation log-loss of sigmoid(logit / T) (grid search; 1 parameter)."""
    logits, y = np.asarray(logits, dtype=float), np.asarray(y)
    best, best_t = np.inf, 1.0
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 200)):
        p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best, best_t = nll, t
    return float(best_t)
