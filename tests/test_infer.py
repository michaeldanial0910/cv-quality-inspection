"""Inference plumbing against a stand-in ONNX model (tests/conftest.py)."""
import json

import numpy as np

from src.infer import Inspector, Policy, overlay
from src.preprocess import to_input


def img(v=128):
    return (np.ones((300, 300)) * v).astype(np.uint8)


def test_logit_equals_head_on_pooled_features(dummy_models):
    ins = Inspector.load(dummy_models)
    logit, feats = ins.raw(to_input(img(90))[None])
    assert np.allclose(logit[0], ins.w @ feats[0].mean(axis=(1, 2)) + ins.b, atol=1e-4)


def test_inspect_result_fields(dummy_models):
    r = Inspector.load(dummy_models).inspect(img())
    assert 0 <= r.probability <= 1 and r.heatmap.shape == (224, 224)
    assert r.zone in ("auto-pass", "human review", "auto-reject")
    json.dumps(r.to_dict())


def test_cam_points_at_the_changed_region(dummy_models):
    """Brighten one corner: if that raises the logit, the CAM must peak in that corner."""
    ins = Inspector.load(dummy_models)
    a = img(128).copy()
    a[:100, :100] = 255
    b = img(128).copy()
    b[:100, :100] = 0
    la, lb = ins.raw(to_input(a)[None])[0][0], ins.raw(to_input(b)[None])[0][0]
    hot = a if la > lb else b
    heat = ins.inspect(hot).heatmap
    assert heat[:70, :70].mean() > heat[150:, 150:].mean()


def test_zones_and_temperature():
    p = Policy(temperature=2.0, review_from=0.2, reject_from=0.8)
    assert [p.zone(x) for x in (0.1, 0.5, 0.9)] == ["auto-pass", "human review", "auto-reject"]


def test_out_of_distribution_goes_to_review(dummy_models, tmp_path):
    ins = Inspector.load(dummy_models)
    ins.ood_ref = np.eye(8, dtype=np.float32)          # reference embeddings unlike anything we will send
    ins.policy = Policy(review_from=0.0, reject_from=0.0, ood_distance=0.01)
    r = ins.inspect(img(200))
    assert r.out_of_distribution and r.zone == "human review"


def test_overlay_size(dummy_models):
    assert overlay(img(), np.zeros((224, 224))).size == (224, 224)
