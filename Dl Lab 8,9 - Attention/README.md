# Experiments 8 & 9 — Effect of Attention Type and Attention Position in a Pretrained ResNet50

**Course:** CS3807 — Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026–27  

## Overview

This notebook implements a systematic ablation study examining how different attention mechanisms (Channel, Spatial, Sequential Channel-Spatial, Self-Attention, ECA, CBAM, and Coordinate Attention) alter an ImageNet-pretrained **ResNet50** backbone, and how the stage at which attention is integrated influences image classification and feature representation on the **TensorFlow Flowers** dataset (`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`). The study combines quantitative performance benchmarking with model complexity metrics, training dynamics, confusion matrix inspection, and qualitative interpretability through internal feature maps and Grad-CAM visualizations.

## Dataset

- **TensorFlow Flowers (`tf_flowers`)** — downloaded and extracted directly in-notebook (`flower_photos.tgz`).
- **Total Images:** 3,670 natural RGB images across 5 classes.
- **Pre-processing:** All images are resized to $224 \times 224 \times 3$. Data augmentation includes random horizontal flips, rotations, zoom, translation, and contrast adjustments.
- **Split:** Fixed 70/15/15 stratified partition:
  - **Train:** 2,569 images
  - **Validation:** 550 images
  - **Test:** 551 images

## Environment / Dependencies

- Google Colab (GPU runtime recommended, e.g., NVIDIA T4)
- TensorFlow / Keras
- scikit-learn (`train_test_split`, `precision_recall_fscore_support`, `confusion_matrix`, `classification_report`)
- NumPy, Pandas, Matplotlib, pathlib, urllib, tarfile

## Key Design Choice: Two-Phase Controlled Transfer Learning

To ensure fair attribution across all 19 model variations, an identical training protocol and dataset split are maintained throughout:
- **Phase 1 (Head Training):** The ResNet50 backbone is frozen; only the newly added attention module and classification head (Global Average Pooling $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense 5-class softmax) are trained for 8 epochs with Adam ($\text{lr} = 10^{-3}$).
- **Phase 2 (Fine-Tuning):** The `conv5_x` stage of ResNet50 is unfrozen (with BatchNorm layers kept frozen) and fine-tuned for 7 epochs using a reduced learning rate ($\text{lr} = 10^{-4}$).
- **Standardized Insertion:** Attention blocks are inserted after residual stages ($P_1$ through $P_4$) matching input/output feature dimensions without disrupting residual dimensionality.

## Notebook Structure

| Section / Experiment | Content |
|---|---|
| **Setup & Preprocessing** | Environment configuration, deterministic seeds, `tf_flowers` download, stratified 70/15/15 split, and tf.data pipeline creation. |
| **ResNet50 Backbone** | Loading ImageNet-pretrained ResNet50 and extracting intermediate activation taps at $P_1$ (`conv2_x`), $P_2$ (`conv3_x`), $P_3$ (`conv4_x`), and $P_4$ (`conv5_x`). |
| **Attention Modules** | Implementations of Channel (SE), Spatial ($7\times7$), Sequential Channel-Spatial, Lightweight Self-Attention, ECA, CBAM, and Coordinate Attention. |
| **Experiment A (Mechanism Ablation)** | Fixed position at $P_3$ comparing Baseline vs. Channel vs. Spatial vs. Channel+Spatial vs. Self-Attention vs. Combined vs. ECA vs. CBAM vs. Coordinate Attention. |
| **Experiment B (Position Ablation)** | Fixed Channel+Spatial module placed at $P_1$ ($56\times56$), $P_2$ ($28\times28$), $P_3$ ($14\times14$), and $P_4$ ($7\times7$), along with multi-stage configurations ($P_3+P_4$, $P_2+P_3+P_4$) and kernel sizes ($k=3, 5, 7$). |
| **Evaluation & Dynamics** | Epoch-by-epoch loss and accuracy curves (marking the unfreeze threshold), parameter audits, GPU inference latency benchmarks, and test metrics. |
| **Interpretability & Visualization** | Confusion matrices, internal channel weight plots, forward spatial attention maps, and post-hoc Grad-CAM activation overlays. |

## Key Results

| Model Architecture | Placement | Test Accuracy | Macro F1 | Attention Params | Inference Time |
|---|---|---|---|---|---|
| **Baseline ResNet50** | None | 93.47% | 0.9345 | 0 | 7.25 ms |
| **Spatial Attention (Winner)** | $P_3$ | **94.19%** | **0.9411** | 99 | 7.44 ms |
| **Channel + Spatial** | $P_3$ | 93.10% | 0.9315 | 132,259 | 6.95 ms |
| **Channel + Spatial** | $P_4$ | 93.47% | 0.9339 | 526,563 | 7.63 ms |
| **Channel (SE)** | $P_3$ | 92.20% | 0.9213 | 132,160 | 8.01 ms |
| **Self-Attention** | $P_3$ | 92.38% | 0.9219 | 1,312,001 | 6.74 ms |
| **Coordinate Attention** | $P_3$ | 92.74% | 0.9282 | 100,512 | 7.65 ms |
| **CBAM** | $P_3$ | 90.02% | 0.8970 | 132,259 | 7.93 ms |
| **ECA** | $P_3$ | 88.57% | 0.8841 | 5 | 7.32 ms |

- **Mechanism Effect:** Spatial Attention at $P_3$ yielded the top test accuracy (94.19%) with an addition of only 99 parameters. Larger modules (Self-Attention with $+1.31\text{M}$ parameters) did not improve generalization.
- **Positional Impact:** Positioning attention at earlier stages ($P_1$: 90.93%, $P_2$: 90.38%) degraded performance relative to later semantic layers ($P_3$: 93.10%, $P_4$: 93.47%). Multi-stage cascade ($P_3+P_4$) reached 93.83% at the cost of $5\times$ more parameters.
- **Interpretability Insights:** Internal spatial attention maps remain largely uniform, whereas Grad-CAM visual overlays confirm that attention sharpens class activation over compact foreground targets (such as dandelion seed heads) while occasionally drifting onto background clutter in misclassified instances.

## How to Run

1. Open the notebook in **Google Colab** and select a GPU runtime (`Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ `T4 GPU`).
2. Execute all cells sequentially (`Runtime` $\rightarrow$ `Run all`).
3. Dataset images are fetched, validated, and extracted automatically into `/content/flower_photos`.
4. Metric dictionaries and summary logs are stored in `/content/results.json`, while plots and Grad-CAM comparisons are saved to `/content/figures/`.
5. Full run time: dataset preprocessing (~1 min) + model training across ablations (~45–60 min on GPU).