"""CPU inference for the fine-tuned model (ONNX Runtime, no torch needed).

    from src.infer import Inspector
    r = Inspector.load().inspect("some_part.jpeg")   # label, probability, zone, heatmap, ood flag

    python -m src.infer path/to/image.jpeg
"""
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CLASS_NAME, MODELS
from src.preprocess import IMG_SIZE, to_gray_image, to_input

ZONES = ("auto-pass", "human review", "auto-reject")


@dataclass
class Policy:
    """Thresholds on the calibrated defect probability, all chosen on validation (src/evaluate.py)."""
    temperature: float = 1.0
    review_from: float = 0.5        # below: auto-pass
    reject_from: float = 0.5        # at/above: auto-reject; in between: a person looks at it
    ood_distance: float = float("inf")

    @classmethod
    def load(cls, path=MODELS / "policy.json"):
        p = Path(path)
        return cls(**{k: v for k, v in json.loads(p.read_text()).items() if k in cls.__annotations__}) if p.exists() else cls()

    def zone(self, prob: float) -> str:
        return ZONES[0] if prob < self.review_from else ZONES[2] if prob >= self.reject_from else ZONES[1]


@dataclass
class Result:
    probability: float              # calibrated P(defective)
    logit: float
    zone: str
    label: str
    heatmap: np.ndarray = field(repr=False)   # (224, 224) in [0, 1]: where the evidence for "defective" is
    ood_distance: float = 0.0
    out_of_distribution: bool = False
    ms: float = 0.0

    def to_dict(self):
        return {"probability": round(self.probability, 4), "logit": round(self.logit, 4), "zone": self.zone,
                "label": self.label, "out_of_distribution": self.out_of_distribution,
                "ood_distance": round(self.ood_distance, 4), "ms": round(self.ms, 1)}


class Inspector:
    def __init__(self, onnx_path, head_path, policy: Policy, ood_ref=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        head = np.load(head_path)
        self.w, self.b = head["w"].astype(np.float32), float(np.ravel(head["b"])[0])
        self.policy = policy
        self.ood_ref = None if ood_ref is None else np.load(ood_ref)["emb"]   # unit-norm training embeddings

    @classmethod
    def load(cls, models=MODELS):
        models = Path(models)
        ood = models / "ood_ref.npz"
        return cls(models / "model.onnx", models / "head.npz", Policy.load(models / "policy.json"),
                   ood if ood.exists() else None)

    def raw(self, batch: np.ndarray):
        """(n, 3, 224, 224) -> logits (n,), feature maps (n, C, h, w)."""
        logit, feats = self.sess.run(["logit", "features"], {"image": batch.astype(np.float32)})
        return logit, feats

    def cam(self, feats: np.ndarray) -> np.ndarray:
        """Class activation map: head-weighted sum of the last feature maps, upsampled to 224."""
        m = np.tensordot(self.w, feats, axes=(0, 0))                     # (h, w)
        m = np.maximum(m, 0)
        m = m / (m.max() + 1e-8)
        return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE),
                                                                            Image.Resampling.BICUBIC)) / 255.0

    def embed(self, feats: np.ndarray) -> np.ndarray:
        e = feats.mean(axis=(-1, -2))
        return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-8)

    def ood_distance(self, emb: np.ndarray) -> float:
        """1 - cosine similarity to the 5th-nearest training image (0 = looks like training data)."""
        if self.ood_ref is None:
            return 0.0
        sims = np.sort(self.ood_ref @ emb)[::-1]
        return float(1 - sims[min(4, len(sims) - 1)])

    def prob(self, logit) -> np.ndarray:
        return 1 / (1 + np.exp(-np.asarray(logit) / self.policy.temperature))

    def inspect(self, img) -> Result:
        t0 = time.perf_counter()
        x = to_input(img)[None]
        logit, feats = self.raw(x)
        ms = (time.perf_counter() - t0) * 1000
        p = float(self.prob(logit[0]))
        d = self.ood_distance(self.embed(feats[0]))
        ood = d > self.policy.ood_distance
        zone = ZONES[1] if ood else self.policy.zone(p)     # unfamiliar input always goes to a person
        return Result(probability=p, logit=float(logit[0]), zone=zone, label=CLASS_NAME[int(p >= 0.5)],
                      heatmap=self.cam(feats[0]), ood_distance=d, out_of_distribution=bool(ood), ms=ms)

    def predict_batch(self, images, bs: int = 32):
        """-> logits (n,), unit embeddings (n, C) for many images."""
        logits, embs = [], []
        for s in range(0, len(images), bs):
            x = np.stack([to_input(i) for i in images[s:s + bs]])
            lg, f = self.raw(x)
            logits.append(lg)
            embs.append(self.embed(f))
        return np.concatenate(logits), np.concatenate(embs)


def overlay(img, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Grey part image with the heatmap in a warm colour on top (for the app, API and reports)."""
    base = to_gray_image(img).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    h = np.clip(heat, 0, 1)[..., None]
    warm = np.array([214, 86, 46], dtype=np.float32)               # same orange as the app's "defect" colour
    arr = np.asarray(base, dtype=np.float32)
    out = arr * (1 - alpha * h) + warm * (alpha * h)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    r = Inspector.load().inspect(sys.argv[1])
    print(json.dumps(r.to_dict(), indent=2))
