# Model card: casting defect classifier (EfficientNet-B0, fine-tuned)

| | |
|---|---|
| **Task** | Binary image classification of top-view photos of cast pump impellers: acceptable vs defective |
| **Output** | Calibrated P(defective), a route (auto-pass / send to a person / auto-reject), a 7×7 class activation heatmap, an out-of-distribution flag |
| **Architecture** | EfficientNet-B0 (ImageNet-1k weights, torchvision) with a new 1-logit head; stages 5–8 fine-tuned |
| **Artefact** | `models/model.onnx` (16 MB, opset 17), outputs `logit` and `features` (1280×7×7); `head.npz`, `policy.json`, `ood_ref.npz` |
| **Runtime** | ONNX Runtime on CPU: 36 ms median, 53 ms p95 per image (4 threads). No GPU needed |
| **Trained** | Colab T4, 1.6 min; torch 2.11, torchvision 0.26; split manifest SHA-256 `8c5f4732…` |
| **Author / date** | Michael Danial, October 2026 (UneeQ Interns, ML track) |

## Intended use

- Decision support on a single production line that photographs this impeller from the top under the lighting seen in training.
- Parts in the "send to a person" zone, and every image flagged as unfamiliar, must be inspected by a person.
- **Not for:** other parts, other camera angles or setups without re-validation; fully automatic accept/reject of
  borderline parts; any commercial use of this specific model (training data is CC BY-NC-ND 4.0).

## Data

Kaggle "Real-life industrial dataset of casting product": 1,300 original 512×512 grayscale photos (60% defective).
Repeat photos of the same part (found with a rotation-invariant polar similarity ≥ 0.97) were kept within one split:
910 train / 195 validation / 195 test. Kaggle's own 300×300 train/test split was **not** used: 707 of its 715 test
images have a near-copy in its train split.

## Performance (test, 195 photos, 117 defective; evaluated once)

| Metric | Value (95% bootstrap CI) |
|---|---|
| PR-AUC | 1.000 |
| ROC-AUC | 1.000 |
| At the threshold set on validation for ≥99% defect recall: defects missed / good parts rejected | 0 / 117 · 11.5% (4.9–19.0%) |
| Three-zone policy: auto-pass / review / auto-reject | 67 (all good) / 14 (3 defective) / 114 (all defective) |
| Accuracy at a 0.5 cut-off (for reference only) | 97.9% (4 defects missed) |
| Expected calibration error after temperature scaling (T = 1.64) | val 0.060, test 0.039 |

Baselines on the same test images: HOG + logistic regression PR-AUC 0.986, 87% of good parts rejected at ≥99% recall;
frozen EfficientNet-B0 + logistic regression PR-AUC 1.000, 15%.

## Explanations

Class activation maps are exact for this head (global average pool + one linear layer). In a deletion test, blurring the
hottest 10% of the map lowered the defect logit by 3.6 vs 1.7 for a random patch (difference CI 1.5–2.4).

## Known limitations and risks

- **Threshold drift:** under simulated lighting, blur, noise, rotation, shift or JPEG changes, defects are still caught (≤ 1 miss)
  but strong changes get 25–100% of good parts flagged (14% on clean photos); mild ones (1 px blur, 10° rotation, light noise) barely matter. Re-validate thresholds after any camera or lighting change, and
  monitor the review rate.
- **Small test set:** 0/117 missed defects only bounds the miss rate below about 2.6% (95%).
- **Hardest cases:** tiny rim pinholes under bright lighting. All 3 defective parts in the review zone look like this.
- **Binary labels, unknown labelling process,** one product and one viewpoint.
- **Non-casting images** get confident scores from the classifier alone. The OOD guard (cosine distance to training
  embeddings, limit = 99th percentile on validation) caught 8/8 such inputs in testing, but it is a heuristic.

## Ethical and safety considerations

A missed defect can reach a customer as a faulty pump. The system is designed so that the model never has the final word on uncertain or unfamiliar parts, and its error profile under drift is conservative (more review, not more escapes). Those properties were measured on simulated drift only and must be re-checked on real production data before any deployment.
