# Experiment 04 - Transfer Learning for Image Classification

## Overview

This project implements **Transfer Learning using the VGG16 Convolutional Neural Network** with **TensorFlow/Keras** for multi-class image classification using the **CIFAR-10 Dataset**. The experiment demonstrates the complete transfer learning workflow, including dataset preparation, loading a pre-trained VGG16 model, feature extraction using frozen layers, model training, fine-tuning, and evaluation.

---

## Objective

- Understand the concept and working of **Transfer Learning**.
- Use a pre-trained **VGG16** model for image classification.
- Perform feature extraction using a pre-trained convolutional base.
- Build a custom classification head for CIFAR-10.
- Train the classification head while keeping the VGG16 base frozen.
- Fine-tune the final convolutional block of VGG16.
- Compare model performance before and after fine-tuning.
- Evaluate the model using accuracy, precision, recall, and F1-score.
- Visualize training performance using learning curves.
- Analyze classification performance using a confusion matrix and misclassified images.

---

## Dataset

**Dataset:** CIFAR-10 Dataset

**Source:** TensorFlow/Keras (`tensorflow.keras.datasets.cifar10`)

**Instances:** 60,000 total images

- 50,000 training images
- 10,000 testing images

**Features:**

- 32 × 32 RGB color images
- 3 color channels
- Input shape: **32 × 32 × 3**
- Pixel values normalized to the range **[0, 1]**

**Classes:**

- 0 – Airplane
- 1 – Automobile
- 2 – Bird
- 3 – Cat
- 4 – Deer
- 5 – Dog
- 6 – Frog
- 7 – Horse
- 8 – Ship
- 9 – Truck

---

## Data Preparation

The original 50,000 training images are divided into:

- **45,000 images** – Training set
- **5,000 images** – Validation set
- **10,000 images** – Test set

A stratified train-validation split with `random_state=42` is used to maintain the class distribution.

The class labels are converted into **one-hot encoded vectors** using `to_categorical()`.

| Dataset | Images | Shape |
|---------|--------|-------|
| Training | 45,000 | 32 × 32 × 3 |
| Validation | 5,000 | 32 × 32 × 3 |
| Testing | 10,000 | 32 × 32 × 3 |

---

## Transfer Learning

The experiment uses the **VGG16 architecture pre-trained on ImageNet**.

The original VGG16 classification head is removed using:

```python
include_top=False