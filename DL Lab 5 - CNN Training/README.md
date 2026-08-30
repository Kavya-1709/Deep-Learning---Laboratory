# Experiment 5 — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

**Course:** CS3807 — Deep Learning Laboratory
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**AY:** 2026–27

## Overview

This notebook implements a full study of how weight initialization, regularization,
optimization algorithms, CNN hyperparameters, transfer learning, and fine-tuning affect
image classification performance, using **MobileNetV2** on the **Oxford-IIIT Pet Dataset**
(37 cat/dog breeds, RGB images resized to 224×224×3). The study concludes with 5-fold
cross-validation for model selection and a final held-out test evaluation.

## Dataset

- **Oxford-IIIT Pet Dataset** — downloaded and extracted directly in-notebook.
- Train+val images: 3,680 · Test images: 3,669 · Classes: 37
- Test set is kept fully untouched until final evaluation (Section 12).

## Environment / Dependencies

- Google Colab (GPU runtime recommended)
- TensorFlow / Keras
- scikit-learn (`StratifiedKFold`, precision/recall/F1, confusion matrix)
- NumPy, Pandas, Matplotlib

## Key Design Choice: Feature Caching

Since MobileNetV2's convolutional base is **frozen** for every section except fine-tuning
(Section 10, Case B), all images are passed through the frozen base **once** early in the
notebook and the resulting 1280-dim feature vectors are cached as NumPy arrays. Every
subsequent experiment (init comparison, regularization, optimizers, hyperparameter sweep,
5-fold CV, overall results, additional exercise) trains only a small Dense head on these
cached features instead of re-running images through the full conv base each time. This
is mathematically identical to training end-to-end on a frozen base, but avoids repeatedly
rebuilding/re-tracing the MobileNetV2 graph, which was causing Colab RAM crashes during the
CV loop (Section 11).

Section 10 (Case B, fine-tuning) is the only section that still uses the raw image
pipeline, since it's the only place the base actually unfreezes and needs real gradients.

## Notebook Structure

| Section | Content |
|---|---|
| 1–3 | Setup, dataset download/extraction, train/val/test split |
| 4 | MobileNetV2 base model load + feature caching |
| 5 | Weight initialization comparison (Zero / Random / Xavier / He) |
| 6 | Regularization comparison (None / L2 / Dropout / BatchNorm) |
| 7 | Batch Normalization — worked numerical example |
| 8 | Optimizer comparison (SGD / Momentum / RMSProp / Adam) |
| 9 | Hyperparameter sweep (learning rate, batch size, dropout rate) |
| 10 | Transfer learning: feature extraction vs. fine-tuning |
| 11 | 5-fold cross-validation across 4 candidate configurations |
| 12 | Final model retrain + test set evaluation |
| 13 | Overall results summary across all stages |
| 14–15 | Plot inferences and discussion questions (see report) |
| 16 | Additional exercise — 2 new configs vs. selected configuration |

## Key Results

| Study | Winner |
|---|---|
| Weight Initialization | **Xavier** (90.49% val. accuracy) |
| Regularization | **BatchNorm** (90.62% val. accuracy) |
| Optimizer | **Adam** (91.17% val. accuracy) |
| Hyperparameters | LR = 0.001, Batch size = 16, Dropout = 0.5 |
| 5-Fold CV | **C1 (Baseline)** — 91.41% ± 0.50% |
| Final Test Accuracy | **89.89%** (Precision 0.9015, Recall 0.8983, F1 0.8979) |
| Additional Exercise | **New_A** config (91.44% ± 1.11%) edges out the selected config |

Most confused pair: American Pit Bull Terrier ↔ Staffordshire Bull Terrier (visually very
similar breeds). Best-classified breeds: Yorkshire Terrier, Scottish Terrier, German
Shorthaired, Saint Bernard, Newfoundland.

## Known Fix Applied

Section 10's fine-tuning step unfreezes the last 30 layers of the shared `base_model`
object. Since Section 12 originally reused that same `base_model` without re-freezing it,
the final model was accidentally trained with a large learning rate on a partially
unfrozen base — collapsing test accuracy to ~40%. Fixed by explicitly setting
`base_model.trainable = False` at the start of Section 12, restoring proper frozen-base
behavior and the expected ~90% test accuracy.

## How to Run

1. Open in Google Colab with a GPU runtime.
2. Run all cells top to bottom (Runtime → Restart runtime first for a clean state).
3. Dataset downloads automatically; no manual upload needed.
4. Full run time: dataset download + feature caching (~5 min) + fine-tuning (~2 min) +
   remaining feature-based sections (a few minutes total).
