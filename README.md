# Casting Quality Inspection with Transfer Learning

> UneeQ Interns · Machine Learning Track · October 2026
> **Demo video:** [`demo/inspection_demo.mp4`](demo/inspection_demo.mp4) (30 s, the Streamlit app) · **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md) · **Author:** Michael Danial

A foundry photographs every cast pump impeller from the top and wants each one classified as **acceptable** or
**defective**. This project builds that classifier with **transfer learning** (EfficientNet-B0 pretrained on ImageNet)
and compares it fairly against a **non-transfer baseline** (HOG features + logistic regression). It uses a
**train/validation/test split that is checked for leakage**, and it does not stop at a label. Every part is routed:
**auto-pass**, **auto-reject**, or **send to a person**. A heatmap shows where the evidence is, and a guard sends
images that don't look like castings to a person instead of guessing. It is served by a FastAPI endpoint and a
five-page Streamlit app.

**Headline results (195 test photos from parts never seen in training, 117 defective):**

| Model | PR-AUC (95% CI) | Good parts rejected to catch ≥99% of defects (95% CI) | Defects missed |
|---|---|---|---|
| A · HOG + logistic regression (no transfer) | 0.986 (0.973–0.995) | 87% (79–94%) | 0 |
| B · frozen EfficientNet-B0 + logistic head | **1.000** | 15% (8–24%) | 0 |
| C · fine-tuned EfficientNet-B0 (**served**) | **1.000** | **12% (5–19%)** | 0 |

With the three-zone policy, the served model put **67 test parts on auto-pass (all good), 114 on auto-reject (all
defective) and 14 in human review (3 defective, 11 good)**. That is zero wrong calls in the automatic zones, with 7%
of parts going to a person.

What came out negative or inconclusive is reported as it came out:

- **B vs C cannot be separated on this test set.** Both rank all 195 photos perfectly. C is served because it was
  best on validation, not because of test.
- **Kaggle's own train/test split is leaky** (§2.2). This project does not use it.
- **Temperature scaling did not improve calibration** on validation (ECE 0.056 → 0.060).
- **Camera or lighting drift breaks the thresholds** (§6). Under strong changes, defects are still caught, but good parts get flagged at 2–7× the normal rate.

---

## 1. Business problem

Two mistakes cost different things. A **missed defect** ships a faulty pump: warranty claims, field failures,
reputation. A **false reject** scraps or re-inspects a good casting, which costs material and labour. So the goal is
not "highest accuracy". It is **catch essentially every defect while throwing away as few good parts as possible**,
and **let a person decide when the model is unsure**. Automatic fraud-style "the model decides" is not acceptable
for borderline parts; the human-review zone is part of the design.

## 2. Data

- **Source:** [Real-life industrial dataset of casting product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
  (Kaggle, ravirajsinh45), licence **CC BY-NC-ND 4.0**. Images are not redistributed in this repo.
- **What's in the zip:** 1,300 original 512×512 grayscale top-view photos (781 defective, 519 acceptable), plus
  7,348 300×300 copies split by the uploader into train (6,633) and test (715).
- **Integrity:** zip SHA-256 `a1123697…0437`, checked by `python -m src.data`.
- **Positive class = defective.**

### 2.1 The originals contain repeat photos of the same part

Some castings were photographed more than once, rotated by an arbitrary angle. If one photo of a part lands in train
and another in test, the model is graded on a part it has already seen. `src/dedupe.py` finds these:

1. Centre each part on its strongest edges and **unwrap it into polar coordinates** (distance from centre × angle).
   A rotation of the part becomes a sideways shift of that strip.
2. Similarity = best normalised correlation over **all shifts and the mirror image** (computed with an FFT). The per-radius
   mean is removed first, so the shape every impeller shares does not count as similarity.
3. Pairs with similarity ≥ **0.97** are the same part. The threshold was set **by eye** from pairs across the range
   (`reports/figures/duplicate_pairs.png`), never from a model score: at ≥ 0.97 the pairs are the same casting, and
   at 0.95 different castings under the same lighting start to merge.

A simpler first attempt (32×32 pixel correlation with 90° turns) failed: every casting looks alike at that size, so
different parts scored above 0.95.

**Result:** 498 of 1,300 photos have at least one near-copy. They form 993 groups (largest 8 photos), and **no group mixes the two labels**.

### 2.2 Kaggle's ready-made split leaks

The same check on the 300×300 version found that **707 of its 715 test images have a near-copy in its train split**,
and **64 test files are byte-identical to a train file**. The 7,348 images are the originals augmented many times,
and copies landed on both sides. Accuracy measured on that split partly measures memorisation. → `reports/kaggle_split_leakage.csv`.

### 2.3 The split used here

| Split | Photos | Distinct parts | Defective | Used for |
|---|---|---|---|---|
| train | 910 | 689 | 60% | fitting |
| val | 195 | 155 | 60% | every choice: epochs, C, thresholds, temperature, OOD limit |
| test | 195 | 149 | 60% | scored once, after everything was fixed |

Groups were assigned whole, stratified by label (`src/split.py`). The manifest `data/split_manifest.csv` (SHA-256
`8c5f4732…`) is committed. `tests/test_split.py` checks that no group and no identical file crosses splits. The Colab
notebook embeds the manifest and checks its hash **and every image's hash** before training, so all three models
train and test on exactly the same files.

## 3. Approach

```
photo ─► grayscale, 224×224, ImageNet normalisation (src/preprocess.py, identical in Colab and serving)
   │
   ├─► A  HOG (128 px, 9 orientations) ─► logistic regression          (CPU, C chosen on val)
   ├─► B  EfficientNet-B0 frozen ─► 1,280 pooled features ─► logistic regression   (C chosen on val)
   └─► C  EfficientNet-B0: new 1-logit head; 4 epochs head only, then blocks 5–8 unfrozen
          (AdamW, cosine LR, early stopping on val PR-AUC, best of 15 epochs) ─► ONNX
                          │
             temperature scaling (val) ─► three zones (val) ─► OOD guard (train embeddings, val limit)
                          │
          auto-pass  ·  send to a person  ·  auto-reject     + CAM heatmap
```

**Augmentation, training images only:** random 90° turns and horizontal flips (the part is round and shot from
directly above, so these are real views), ±15° rotation, brightness/contrast jitter (±20%), light blur. **No random
crops**: a crop can cut off a rim defect and leave a "defective" label on a clean-looking image.

**Why EfficientNet-B0:** small (16 MB ONNX, ~36 ms per image on a laptop-class CPU), strong ImageNet features, and a
global-average-pool + single linear head, which makes the class activation map exact (§5).

**Fair comparison:** same split, same test rows, same evaluation code (`src/evaluate.py` asserts the test rows are
identical), and every hyper-parameter chosen on validation.

## 4. Results

### 4.1 Model comparison

| Test set | A · HOG + LR | B · frozen | C · fine-tuned |
|---|---|---|---|
| PR-AUC | 0.986 | 1.000 | 1.000 |
| ROC-AUC | 0.977 | 1.000 | 1.000 |
| Threshold for ≥99% defect recall (set on val) | 0.0001 | 0.044 | 0.006 |
| Defects missed at that threshold | 0 / 117 | 0 / 117 | 0 / 117 |
| Good parts rejected at that threshold | 87.2% | 15.4% | 11.5% |
| Accuracy at a plain 0.5 cut-off | 91.3% | 100.0% | 97.9% |

- **Transfer learning clearly helps:** paired on the same images, PR-AUC C − A = **+0.014 (95% CI +0.005 to +0.027)**.
  The practical gain is bigger than that number suggests: at the 99%-recall operating point, HOG scraps 87% of good parts and the CNNs scrap 12–15%.
- **B vs C:** PR-AUC difference exactly 0. On validation C was ahead (AP 0.998 vs 0.995). The test set is too small and
  too easy to separate them. The frozen model is a perfectly defensible choice too.
- **The thresholds are tiny** (0.006 for C) because the validation set contains a few hard defects with low scores, and
  catching 99% of defects means setting the bar under them. That is what pushes good parts into the reject pile.

### 4.2 The decision policy (served model, chosen on validation)

| Zone | Rule | Test parts | Defective | Good |
|---|---|---|---|---|
| Auto-pass | P(defect) < 0.038 (keeps 99.5% of val defects above) | 67 | 0 | 67 |
| Send to a person | in between | 14 | 3 | 11 |
| Auto-reject | P(defect) ≥ 0.239 (val precision ≥ 99%) | 114 | 114 | 0 |

Probabilities are temperature-scaled (T = 1.64, fitted on validation logits). Calibration did not improve on
validation (ECE 0.056 → 0.060; test 0.039), so the zone edges were set directly on validation rather than relying on "0.5 means 50%".

### 4.3 Training

Fine-tuning took 1.6 minutes on a Colab T4. Validation loss was lowest at fine-tune epoch 6. The checkpoint rule
(best validation PR-AUC, fixed before training) picked epoch 11 (val loss 0.171 vs 0.136). The rule is reported,
not changed after seeing results. ONNX export matches PyTorch to 0.00013 in logit on all 195 validation images.

## 5. Misclassification analysis and explainability

**What went wrong on test:** at the served policy, nothing was wrongly auto-passed or auto-rejected. The 14 parts sent
to a person (`reports/errors_test.csv`, `reports/figures/errors_review.png`):

| Group | n | P(defect) | What they have in common |
|---|---|---|---|
| Defective | 3 | 0.14–0.15 | a **tiny pinhole or dent on the outer rim**, photographed against the **brighter background** setup |
| Good | 11 | 0.04–0.12 | **machining marks / surface texture on the rim**, no visible defect; faint, scattered heatmaps |

At a plain 0.5 cut-off the fine-tuned model would have missed 4 defects (96.6% recall). That is why a single 0.5
threshold is the wrong tool here.

**Heatmaps (class activation maps):** because the network ends in global average pooling + one linear layer, the
defect logit is exactly the average over the 7×7 grid of `w · features`. So `w · features` per cell (the CAM) is
an exact map of which regions added to the score. It is computed from the ONNX outputs; no gradients are needed.

**Do the heatmaps mean anything? Deletion test** (`src/explain.py`): for the 117 defective test parts the model
flagged, blurring the **hottest 10%** of the heatmap lowered the defect logit by **3.6** on average. Blurring a
**random** patch of the same area lowered it by **1.7**. The difference has a 95% CI of 1.5–2.4, and the hot region
mattered more in 79% of parts. On correctly rejected parts the heatmap sits on the visible rim defect (`reports/figures/heatmap_examples.png`).

## 6. Deployment limitations (measured)

**Camera/lighting drift** (`src/robustness.py`): every test photo was perturbed and re-scored with the thresholds
frozen. The model **fails safe**. Mild changes (1 px blur, 10° rotation, light noise) barely matter. Across all 16 perturbations at most 1 defect was missed, and PR-AUC stayed ≥ 0.98.
But good parts flagged (review or reject) jump from **14% on clean photos** to:

| Change | Good parts flagged |
|---|---|
| brightness ×0.6 / ×1.5 | 63% / 50% |
| blur 2 px / 4 px | 38% / 100% |
| noise σ 25 | 97% |
| rotation 45° | 69% |
| shift 15% of width | 64% |
| JPEG quality 10 | 77% |

The ranking survives and the **thresholds** do not. In production: monitor the review rate as a drift alarm, and
re-set thresholds on fresh labelled photos after any camera or lighting change.

**Inputs that aren't castings** (`reports/ood_check.csv`): a 2-class model always answers "acceptable" or "defective".
Random noise scored P(defect) = 1.00 and a photo of coins 0.50. The guard compares each image's 1,280-d embedding to
the training photos (cosine distance to the 5th-nearest). The limit is the 99th percentile on validation, so 1% of
real parts would be sent to review by it. It routed **8 of 8** non-casting inputs to a person.

**Other limits:**
- **One product, one view, one camera,** under at least two lighting setups (grey and bright backgrounds). A new part or angle needs new data.
- **Small test set:** 0 misses out of 117 defects bounds the true miss rate below ~2.6% (rule of three). It does not prove 99.9%.
- **Binary labels only:** no defect type or location, so no per-type error rates and no segmentation.
- **Labelling process unknown:** the dataset does not document who labelled the parts.
- **Licence:** non-commercial, no derivatives. Fine for this study, not for a product.
- **Speed is not the bottleneck:** 36 ms median per part (p95 53 ms) on 4 CPU threads; 16 MB model; no GPU.

## 7. Inference demo and API

```bash
streamlit run app.py                    # http://localhost:8501
uvicorn api.main:app --reload           # http://localhost:8000/docs
curl -F "file=@part.jpeg" "http://localhost:8000/inspect?heatmap=true"
python -m src.infer path/to/part.jpeg   # command line
```

`POST /inspect` returns the calibrated probability, the zone, the OOD flag and distance, the model time and
(optionally) a base64 PNG heatmap overlay. Unreadable files return 415, and files over 10 MB return 413.

| App page | What it shows |
|---|---|
| **Inspect a part** | pick a test part or upload a photo → route, probability on the zone scale, heatmap; sliders simulate lighting/blur drift live |
| **Results** | A vs B vs C with intervals, the three-zone outcome, a threshold slider (shipped defects vs scrapped parts), the training curve |
| **Review queue** | the 14 test parts a person would see, with heatmaps and tags; the deletion test |
| **Limits** | robustness to drift, the OOD check, known limitations |
| **The data** | the leak in Kaggle's split, how duplicates were found, the split used |

## 8. How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# data (needs a Kaggle account: ~/.kaggle/kaggle.json)
kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product -p data/raw

python -m src.data          # verify the zip checksum, unzip, inventory
python -m src.dedupe        # duplicate groups + Kaggle split leakage check (~80 s)
python -m src.split         # data/split_manifest.csv (committed)
python -m src.baseline      # model A
python notebooks/make_colab_notebook.py
#   → open notebooks/02_train_colab.ipynb in Colab, T4 GPU, Run all (~5–10 min);
#     unzip the downloaded cv_outputs.zip in the repo root (models B and C)
python -m src.evaluate      # A vs B vs C, policy.json, calibration
python -m src.explain       # OOD guard, error galleries, deletion test
python -m src.robustness    # drift suite, latency
pytest -q                   # 29 tests (app tests need the model and test images)
```

The trained model (`models/model.onnx`, 16 MB), its head weights and the OOD reference are committed, so the app and
API work without retraining. Only the images need downloading.

## 9. Repository layout

```
src/        config · data (checksum, unzip) · dedupe (polar similarity) · split · preprocess (shared)
            baseline (HOG+LR) · metrics · evaluate · infer (ONNX, CAM, OOD, zones) · explain · robustness
notebooks/  make_colab_notebook.py → 02_train_colab.ipynb (models B and C, run on Colab)
api/        FastAPI service
app.py, app_pages/, app_lib/   Streamlit app (5 pages)
models/     model.onnx · head.npz · ood_ref.npz · policy.json · colab_meta.json
reports/    metrics, predictions, error tables, robustness, figures
data/       split_manifest.csv (images not committed)
tests/      split leakage · metrics · inference · API · app
demo/       inspection_demo.mp4 + the scripts that recorded and edited it
```

## 10. Next steps

1. **More defective examples and defect-type labels**, then segmentation (where is the defect, how big), which would
   make the heatmaps checkable against ground truth.
2. **A larger, harder test set** from new production days, so B vs C and a 99.9% recall target can actually be tested.
3. **Train with the drift seen in §6** (stronger brightness, blur and JPEG augmentation) and re-measure, then add
   review-rate monitoring with an automatic "re-calibrate" alert.
4. Widen the linear probe's C grid: the best C (1.0) was at the edge of the searched range.
