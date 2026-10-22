import numpy as np

from src.metrics import at_threshold, ece, fit_temperature, threshold_for_precision, threshold_for_recall
from src.preprocess import to_input


def test_confusion_counts():
    r = at_threshold([1, 1, 0, 0], [0.9, 0.2, 0.6, 0.1], 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["defect_recall"] == 0.5 and r["false_reject_rate"] == 0.5


def test_threshold_for_recall_meets_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    t = threshold_for_recall(y, s, 0.99)
    assert at_threshold(y, s, t)["defect_recall"] >= 0.99
    assert at_threshold(y, s, t + 1e-6)["defect_recall"] < 0.99 or t >= s[y == 1].min()


def test_threshold_for_precision():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.7, 0.6, 0.8, 0.9])
    assert threshold_for_precision(y, s, 1.0) == 0.8


def test_ece_perfect_and_bad():
    assert ece([0, 1], [0.0, 1.0]) == 0
    assert ece([0, 0, 0, 0], [0.95] * 4) > 0.9


def test_temperature_shrinks_overconfident_logits():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    true_logit = (y * 2 - 1) * 1.0 + rng.normal(0, 1, 2000)
    assert fit_temperature(true_logit * 8, y) > 2.5


def test_preprocess_shape_and_normalisation():
    img = (np.ones((300, 300)) * 128).astype(np.uint8)
    x = to_input(img)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    assert np.allclose(x[0], (128 / 255 - 0.485) / 0.229, atol=1e-5)
