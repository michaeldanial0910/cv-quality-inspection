"""Explain and stress-check the served model (C) on CPU via ONNX.

1. Out-of-distribution reference: training-image embeddings + a distance threshold set on
   validation (99th percentile), so unfamiliar inputs go to a person, not to a verdict.
2. Error gallery: every test image the policy gets wrong or sends to review, with its heatmap.
3. Deletion test: does blanking the heatmap's hottest region lower the defect score more than
   blanking a random region of the same size? If not, the heatmaps are decoration.

    python -m src.explain
"""
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from skimage import data as skdata

from src.config import EXTRACTED, FIGURES, MANIFEST, MODELS, REPORTS, SEED
from src.infer import Inspector, overlay
from src.preprocess import IMG_SIZE, to_gray_image, to_input

INK, SOFT = (28, 35, 51), (90, 100, 120)


def ood_reference(ins: Inspector, m: pd.DataFrame) -> dict:
    tr = m[m["split"] == "train"]
    _, emb = ins.predict_batch([EXTRACTED / p for p in tr["path"]])
    np.savez_compressed(MODELS / "ood_ref.npz", emb=emb.astype(np.float32))
    ins.ood_ref = emb
    _, ev = ins.predict_batch([EXTRACTED / p for p in m.loc[m["split"] == "val", "path"]])
    dist_val = np.array([ins.ood_distance(e) for e in ev])
    thr = float(np.percentile(dist_val, 99))
    # inputs that are clearly not a casting top view
    rng = np.random.default_rng(SEED)
    foreign = {"random noise": (rng.random((300, 300)) * 255).astype(np.uint8),
               "blank grey": np.full((300, 300), 128, np.uint8),
               "camera (photo)": skdata.camera(), "coins (photo)": skdata.coins(),
               "astronaut (photo)": skdata.astronaut(), "coffee (photo)": skdata.coffee(),
               "text page": skdata.page(), "checkerboard": skdata.checkerboard()}
    rows = []
    for name, im in foreign.items():
        _, e = ins.predict_batch([im])
        d = ins.ood_distance(e[0])
        p = float(ins.prob(ins.raw(to_input(im)[None])[0])[0])
        rows.append({"input": name, "ood_distance": d, "flagged": d > thr, "defect_prob_without_guard": p})
    pd.DataFrame(rows).to_csv(REPORTS / "ood_check.csv", index=False)
    policy = json.loads((MODELS / "policy.json").read_text())
    policy["ood_distance"] = thr
    (MODELS / "policy.json").write_text(json.dumps(policy, indent=2))
    ins.policy.ood_distance = thr
    return {"threshold": thr, "val_flagged": float((dist_val > thr).mean()),
            "foreign_flagged": f"{sum(r['flagged'] for r in rows)}/{len(rows)}"}


def gallery(rows: pd.DataFrame, ins: Inspector, fn, title: str, per_row: int = 4, s: int = 230):
    if rows.empty:
        return
    n = len(rows)
    cols = min(per_row, n)
    nrows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * (2 * s + 24), nrows * (s + 58) + 40), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), title, fill=INK)
    for k, r in enumerate(rows.itertuples()):
        x0, y0 = (k % cols) * (2 * s + 24), 40 + (k // cols) * (s + 58)
        res = ins.inspect(EXTRACTED / r.path)
        sheet.paste(to_gray_image(EXTRACTED / r.path).resize((s, s)).convert("RGB"), (x0 + 6, y0))
        sheet.paste(overlay(EXTRACTED / r.path, res.heatmap).resize((s, s)), (x0 + s + 10, y0))
        truth = "defective" if r.label == 1 else "acceptable"
        d.text((x0 + 6, y0 + s + 4), f"#{k + 1}  truth: {truth}   P(defect) = {r.prob:.3f}", fill=INK)
        d.text((x0 + 6, y0 + s + 22), f"{r.zone}   {r.path.split('/')[-1]}", fill=SOFT)
    sheet.save(fn)


def deletion_test(ins: Inspector, paths, frac: float = 0.1, n_random: int = 5) -> pd.DataFrame:
    """Blur the top-`frac` heatmap pixels vs random pixel blocks of the same area; compare logit drops."""
    rng = np.random.default_rng(SEED)
    rows = []
    for p in paths:
        img = to_gray_image(EXTRACTED / p).resize((IMG_SIZE, IMG_SIZE))
        blurred = img.filter(ImageFilter.GaussianBlur(12))
        res = ins.inspect(img)
        base = res.logit
        k = int(frac * IMG_SIZE * IMG_SIZE)
        top = np.zeros(IMG_SIZE * IMG_SIZE, bool)
        top[np.argsort(res.heatmap.ravel())[::-1][:k]] = True

        def masked_logit(mask):
            out = np.where(mask.reshape(IMG_SIZE, IMG_SIZE), np.asarray(blurred), np.asarray(img)).astype(np.uint8)
            return ins.inspect(out).logit

        drop_cam = base - masked_logit(top)
        side = int(np.sqrt(k))
        drops = []
        for _ in range(n_random):
            m = np.zeros((IMG_SIZE, IMG_SIZE), bool)
            y, x = rng.integers(0, IMG_SIZE - side, 2)
            m[y:y + side, x:x + side] = True
            drops.append(base - masked_logit(m.ravel()))
        rows.append({"path": p, "logit": base, "drop_heatmap": drop_cam, "drop_random": float(np.mean(drops))})
    return pd.DataFrame(rows)


def main():
    ins = Inspector.load()
    m = pd.read_csv(MANIFEST)
    summary = {"ood": ood_reference(ins, m)}

    pred = pd.read_csv(REPORTS / "predictions" / "finetuned_calibrated.csv")
    test = pred[pred["split"] == "test"].copy()
    pol = ins.policy
    test["zone"] = [pol.zone(p) for p in test["prob"]]
    test["error"] = np.select([(test.zone == "auto-pass") & (test.label == 1), (test.zone == "auto-reject") & (test.label == 0),
                               test.zone == "human review"], ["missed defect", "false reject", "sent to review"], "")
    errs = test[test["error"] != ""].sort_values(["error", "prob"])
    errs.assign(tag="").to_csv(REPORTS / "errors_test.csv", index=False)
    for kind, fname in [("missed defect", "errors_missed.png"), ("false reject", "errors_false_reject.png"),
                        ("sent to review", "errors_review.png")]:
        gallery(errs[errs.error == kind].head(16), ins, FIGURES / fname, f"Test images: {kind} ({(errs.error == kind).sum()})")
    ok = test[(test.error == "")]
    gallery(pd.concat([ok[ok.label == 1].sort_values("prob").tail(4), ok[ok.label == 0].sort_values("prob").head(4)]),
            ins, FIGURES / "heatmap_examples.png", "Correct calls: where the model looks")
    summary["errors"] = errs["error"].value_counts().to_dict()

    caught = test[(test.label == 1) & (test.zone != "auto-pass")]["path"]
    dt = deletion_test(ins, caught.tolist())
    dt.to_csv(REPORTS / "deletion_test.csv", index=False)
    diff = (dt["drop_heatmap"] - dt["drop_random"]).values
    rng = np.random.default_rng(SEED)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    summary["deletion_test"] = {"n": int(len(dt)), "mean_logit_drop_heatmap": float(dt.drop_heatmap.mean()),
                                "mean_logit_drop_random": float(dt.drop_random.mean()),
                                "difference_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                "share_heatmap_bigger": float((diff > 0).mean())}
    (REPORTS / "explain_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
