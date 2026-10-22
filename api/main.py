"""HTTP API around the fine-tuned model (ONNX Runtime, CPU).

    uvicorn api.main:app --reload      ->  http://localhost:8000/docs

    curl -F "file=@part.jpeg" "http://localhost:8000/inspect?heatmap=true"
"""
import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.config import MODELS
from src.infer import Inspector, overlay

MODEL_DIR = Path(os.environ.get("CVQI_MODELS", MODELS))
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Casting quality inspection", version="1.0",
              description="EfficientNet-B0 fine-tuned on top-view casting photos. Returns a defect probability and a "
                          "routing decision (auto-pass / human review / auto-reject). It never makes the final call on "
                          "an unfamiliar image: those always go to human review.")


@lru_cache(maxsize=1)
def inspector() -> Inspector:
    return Inspector.load(MODEL_DIR)


class Inspection(BaseModel):
    probability: float = Field(description="Calibrated P(defective)")
    logit: float
    zone: str = Field(description="auto-pass | human review | auto-reject")
    label: str = Field(description="acceptable | defective at a 0.5 cut-off (for reference; use `zone` to act)")
    out_of_distribution: bool = Field(description="True if the image does not look like the training photos")
    ood_distance: float
    ms: float = Field(description="Model time on this server, milliseconds")
    heatmap_png: str | None = Field(None, description="Base64 PNG: the part with the defect heatmap overlaid")


@app.get("/health")
def health():
    ins = inspector()
    meta = MODEL_DIR / "colab_meta.json"
    policy = {k: (None if isinstance(v, float) and v == float("inf") else v) for k, v in ins.policy.__dict__.items()}
    return {"status": "ok", "policy": policy,
            "model": json.loads(meta.read_text()) if meta.exists() else None}


@app.post("/inspect", response_model=Inspection)
async def inspect(file: UploadFile = File(...), heatmap: bool = Query(False)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "not a readable image")
    r = inspector().inspect(img)
    out = r.to_dict()
    if heatmap:
        buf = io.BytesIO()
        overlay(img, r.heatmap).save(buf, "PNG")
        out["heatmap_png"] = base64.b64encode(buf.getvalue()).decode()
    return out
