"""API contract against the stand-in ONNX model."""
import importlib
import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def client(dummy_models, monkeypatch):
    monkeypatch.setenv("CVQI_MODELS", str(dummy_models))
    import api.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def png(v=120):
    b = io.BytesIO()
    Image.fromarray((np.ones((300, 300)) * v).astype(np.uint8)).save(b, "PNG")
    return b.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_inspect_returns_zone_and_heatmap(client):
    r = client.post("/inspect?heatmap=true", files={"file": ("part.png", png(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["zone"] in ("auto-pass", "human review", "auto-reject") and 0 <= j["probability"] <= 1
    assert j["heatmap_png"] and len(j["heatmap_png"]) > 100


def test_rejects_non_images(client):
    r = client.post("/inspect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
