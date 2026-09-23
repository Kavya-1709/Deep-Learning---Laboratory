# Experiment 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Course:** CS3807 — Deep Learning Laboratory
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**AY:** 2026–27

## Overview

This notebook implements an end-to-end comparative study of recurrent neural network architectures (**SimpleRNN**, **LSTM**, and **GRU**) for temporal sequence modeling and classification. Using raw multichannel inertial time-series signals from the **UCI Human Activity Recognition (HAR) Dataset** (6 activity classes, 9 sensor channels across 128 time steps), the study evaluates vanishing gradient dynamics via Backpropagation Through Time (BPTT), compares training and convergence behavior across architectures, and analyzes classification performance through confusion matrices.

## Dataset

- **UCI Human Activity Recognition (HAR) Dataset** — downloaded and extracted directly in-notebook.
- **Signals:** 9 raw tri-axial inertial channels (`total_acc_{x,y,z}`, `body_acc_{x,y,z}`, `body_gyro_{x,y,z}`).
- **Subset:** Balanced stratified subset of 3,000 samples (500 windows per class).
- **Split:** 70/15/15 train/val/test split (Train: 2,100 · Val: 450 · Test: 450) with standard normalization fit strictly on training data.

## Environment / Dependencies

- Google Colab (GPU runtime recommended)
- TensorFlow / Keras
- scikit-learn (`train_test_split`, `StandardScaler`, confusion matrix, precision/recall/F1)
- NumPy, Pandas, Matplotlib, Seaborn, OpenCV (`opencv-python-headless`)

## Key Design Choice: Standardized Recurrent Architecture

To ensure a fair, apples-to-apples comparison across sequence models, **SimpleRNN**, **LSTM**, and **GRU** are built with identical capacity and trained under an identical protocol:
- Input shape of `(128, 9)`, 32 recurrent units, dropout rate of 0.2, a 16-unit Dense layer (ReLU), and a 6-unit Dense output layer (Softmax).
- Trained using the **Adam** optimizer (`lr = 0.001`), `sparse_categorical_crossentropy` loss, batch size of 32, and 30 epochs.
- This isolates the recurrent cell gating mechanism as the sole variable, allowing a direct comparison of parameter counts, wall-clock training times, convergence curves, and final accuracies.

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Setup, library imports, and bold plotting utility configuration |
| 2 | UCI HAR dataset download, signal extraction, stratified subsetting, and train/val/test split |
| 3 | Temporal data visualization — multi-channel sensor signals vs. time step across activities (Plot 1) |
| 4 | Backpropagation Through Time (BPTT) — manual numerical calculation and Python verification |
| 5 | Model builder — functional API construction for SimpleRNN, LSTM, and GRU |
| 6 | Model training — 30-epoch standardized training loop tracking wall-clock time and loss/accuracy histories |
| 7 | Training dynamics visualization — loss curves (Plot 2) and validation accuracy curves (Plot 3) |
| 8 | Model evaluation, parameter comparison, and confusion matrix analysis |
| 9–10 | Bidirectional recurrent variants and video activity sequence understanding extensions |
| 11 | Overall results summary, architectural trade-offs, and report discussion questions |

## Key Results

| Architecture | Parameters | Training Time | Final Val. Accuracy |
|---|---|---|---|
| **SimpleRNN** | 1,974 | ~37.1s | 83.33% |
| **LSTM** | 6,006 | ~28.6s | 94.00% |
| **GRU** | **4,758** | **~27.5s** | **95.11% (Winner)** |

- **BPTT Numerical Verification:** Hand-calculated hidden states ($h_1 \approx 0.3364$, $h_2 \approx 0.6161$, $h_3 \approx 0.5997$) match numerical forward propagation output ($0.3364$, $0.6164$, $0.6000$) within floating-point precision.
- **Observations:** GRU achieved the highest validation accuracy (95.11%) with ~21% fewer parameters and faster epoch convergence than LSTM. SimpleRNN showed slower convergence and lower accuracy (83.33%) due to vanishing gradients across the 128 time steps.

## How to Run

1. Open in Google Colab with a GPU runtime.
2. Run all cells top to bottom (Runtime → Restart session first for a clean state).
3. Dataset downloads and extracts automatically (`har_data/`); no manual upload needed.
4. Generated figures are automatically saved to `plots/`.
5. Full run time: dataset download (~1 min) + training all 3 models (~2 min).