# Experiment 7 — Autoencoders, Convolutional AE, Denoising AE and Variational AE

**Course:** CS3807 — Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026–27

## Overview

This notebook provides an end-to-end study of different autoencoder architectures for image reconstruction and generation. It covers a **Fully Connected Autoencoder, Convolutional Autoencoder, Denoising Convolutional Autoencoder, and Variational Autoencoder (VAE)** using the MNIST dataset. The study evaluates reconstruction quality using MSE, MAE and SSIM, investigates latent-dimension effects, denoising performance, VAE latent spaces, image generation, interpolation, and the effect of KL-loss weighting.

## Dataset

- **MNIST handwritten digit dataset** — loaded directly using Keras.
- **Training subset:** 10,000 images.
- **Test subset:** 2,000 images.
- **Image size:** 28 × 28 grayscale.
- **Pixel normalization:** Pixel values scaled to the range 0–1.
- **Input formats:** 28 × 28 × 1 for convolutional models and 784-dimensional vectors for the fully connected model.
- A fixed random seed of **42** is used for reproducibility.

## Environment / Dependencies

- Google Colab (GPU runtime recommended)
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-image (`structural_similarity`)
- Python standard libraries: `os`, `time`, `gc`, `random`

## Key Design Choices

### Fully Connected Autoencoder

The fully connected autoencoder uses the architecture:

`784 → 128 → 32 → 16 → 32 → 128 → 784`

- Latent dimension: 16
- ReLU hidden layers and sigmoid output
- Adam optimizer with learning rate 0.001
- Binary cross-entropy loss
- Batch size: 128
- Epochs: 20

### Convolutional Autoencoder

The convolutional autoencoder uses convolution and pooling layers to preserve spatial information:

`Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → UpSampling → Conv2D → UpSampling → Conv2D`

- Adam optimizer with learning rate 0.001
- Binary cross-entropy loss
- Batch size: 128
- Epochs: 20

### Denoising Autoencoder

The convolutional architecture is trained to reconstruct clean images from corrupted inputs using:

- Gaussian noise with sigma values 0.1, 0.2 and 0.3
- Salt-and-pepper noise with probabilities 0.05, 0.10 and 0.20
- Batch size: 128
- Epochs: 15

### Variational Autoencoder

The VAE uses a **2-dimensional latent space** for direct visualization.

- Encoder: convolutional layers followed by latent mean and log-variance
- Sampling layer using the reparameterization trick
- Decoder based on transposed convolutions
- Adam optimizer with learning rate 0.001
- Batch size: 128
- Epochs: 30

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Setup, imports, random seed and plotting configuration |
| 2 | MNIST dataset loading, subsetting and preprocessing |
| 3 | Metric utilities and reusable plotting functions |
| 4 | Fully Connected Autoencoder architecture, training and reconstruction |
| 5 | Convolutional Autoencoder architecture, training and reconstruction |
| 6 | Comparison of Fully Connected and Convolutional Autoencoders |
| 7 | Denoising CAE using Gaussian and salt-and-pepper noise |
| 8 | Variational Autoencoder, latent-space visualization, generation and interpolation |
| 9 | Reconstruction-error distribution and high-error samples |
| 10 | Consolidated results of all main models |
| 11 | Fully Connected AE latent-dimension study |
| 12 | Discussion and analysis |
| 13–13.8 | Additional experiments: CAE latent dimensions, denoising, noise levels, decoder comparison, VAE latent dimensions, KL weighting, 100 generated samples and interpolation |

## Key Results

### Main Model Comparison

| Model | MSE | MAE | SSIM | Parameters | Time |
|---|---:|---:|---:|---:|---:|
| **Fully Connected AE** | 0.01996 | 0.05433 | 0.7733 | 211040 | 13.5 s |
| **Convolutional AE** | 0.00248 | 0.01506 | 0.9748 | 74497 | 24.2 s |
| **Denoising CAE** | 0.00460 | 0.02141 | 0.9467 | 74497 | 16.2 s |
| **VAE** | 0.04492 | 0.10270 | 0.5058 | 134165 | 31.5 s |

- The Convolutional AE achieves substantially lower reconstruction error and higher SSIM than the Fully Connected AE.
- The Denoising CAE successfully reconstructs clean images from noisy inputs.
- The VAE provides a continuous 2-D latent space and supports random generation and latent interpolation.

### VAE Loss Breakdown

| Metric | Value |
|---|---:|
| Reconstruction Loss | 152.362 |
| KL Loss | 5.692 |
| Total Loss | 158.054 |
| Test MSE | 0.04492 |
| Test MAE | 0.10270 |
| Mean SSIM | 0.5058 |
| Parameters | 134165 |

### Latent-Dimension Study — Fully Connected AE

| Latent Dimension | MSE | SSIM | Parameters |
|---:|---:|---:|---:|
| 2 | 0.05012 | 0.4338 | 210130 |
| 8 | 0.02792 | 0.6936 | 210520 |
| 16 | 0.02287 | 0.7384 | 211040 |
| 32 | 0.02074 | 0.7615 | 212080 |

Increasing the latent dimension improves reconstruction quality, but the improvement becomes smaller at larger dimensions.

## Denoising Results

### Gaussian Noise

| Noise Level | MSE | MAE | SSIM |
|---:|---:|---:|---:|
| 0.1 | 0.00352 | 0.01847 | 0.9620 |
| 0.2 | 0.00460 | 0.02141 | 0.9467 |
| 0.3 | 0.00691 | 0.02733 | 0.9172 |

### Salt-and-Pepper Noise

| Noise Level | MSE | MAE | SSIM |
|---:|---:|---:|---:|
| 0.05 | 0.00392 | 0.01901 | 0.9581 |
| 0.10 | 0.00512 | 0.02214 | 0.9439 |
| 0.20 | 0.00789 | 0.02855 | 0.9115 |

Increasing noise severity increases reconstruction error and reduces SSIM for both noise types.

## Additional Exercise Results

### Convolutional AE Latent-Dimension Study

| Latent Dimension | MSE | MAE | SSIM | Parameters |
|---:|---:|---:|---:|---:|
| 4 | 0.04214 | 0.09814 | 0.5471 | 65797 |
| 8 | 0.02854 | 0.07092 | 0.6978 | 90889 |
| 16 | 0.02194 | 0.05784 | 0.7668 | 141073 |
| 32 | 0.01267 | 0.03883 | 0.8654 | 241441 |

### Decoder Comparison

| Decoder Type | MSE | MAE | SSIM | Parameters |
|---|---:|---:|---:|---:|
| UpSampling2D | 0.00248 | 0.01506 | 0.9748 | 74497 |
| Conv2DTranspose | 0.00263 | 0.01529 | 0.9740 | 74497 |

### VAE Latent-Dimension Comparison

| Latent Dimension | Reconstruction Loss | KL Loss | MSE | SSIM |
|---:|---:|---:|---:|---:|
| 2 | 152.362 | 5.692 | 0.04492 | 0.5058 |
| 8 | 119.552 | 12.660 | 0.03040 | 0.6830 |

### Beta-VAE

| Beta | Reconstruction Loss | KL Loss | MSE | SSIM |
|---:|---:|---:|---:|---:|
| 1.0 | 160.572 | 5.730 | 0.04877 | 0.4578 |
| 5.0 | 182.508 | 8.093 | 0.05792 | 0.3257 |

Increasing the KL-loss weight produces a more constrained latent representation but reduces reconstruction quality.

### VAE Generation

- 25 random digits are generated using samples from a standard normal latent distribution.
- A larger grid of **100 generated samples** is also produced.
- Mean pairwise pixel MAE across 30 generated samples: **0.1003**.
- Latent-space interpolation is performed between different digit pairs to observe smooth transitions.

## Reconstruction Error Analysis

For the Convolutional AE:

- Error range: **0.00027 to 0.00922**
- Mean error: **0.00248**
- Standard deviation: **0.00127**
- Most images have low reconstruction error.
- A small number of high-error images show visibly poorer or blurrier reconstructions.

## Generated Plots

The notebook automatically saves figures to the `plots/` directory, including:

- `plot1` — Fully Connected AE reconstruction
- `plot2` — FC-AE training and validation loss
- `plot3` — FC-AE vs. Conv-AE reconstruction
- `plot4b_denoising_salt_pepper.png` — Salt-and-pepper denoising
- `plot5_noise_level_vs_quality.png` — Gaussian noise analysis
- `plot5b_saltpepper_noise_level_vs_quality.png` — Salt-and-pepper noise analysis
- `plot6_vae_latent_space.png` — VAE latent space
- `plot7_vae_generated_samples.png` — Random VAE samples
- `plot8_vae_latent_interpolation.png` — Latent interpolation
- `plot9_vae_loss_curve.png` — VAE training and validation loss
- `plot10_reconstruction_error_distribution.png` — Reconstruction-error distribution
- `plot_highest_error_samples.png` — Highest-error samples
- `plot_latent_dim_vs_mse.png` — FC-AE latent-dimension study
- `ex1_cae_latent_dim_sweep.png` — CAE latent-dimension study
- `ex2_gaussian_vs_saltpepper.png` — Denoising comparison
- `ex4_upsampling_vs_transpose.png` — Decoder comparison
- `ex5_vae_latent8_recon.png` — VAE with latent dimension 8
- `ex6_beta_vae_latent_space.png` — Beta-VAE latent spaces
- `ex7_vae_generated_100.png` — 100 generated VAE samples
- `ex8_latent_interpolation_pair2.png` — Second latent interpolation

## How to Run

1. Open the notebook in Google Colab with a GPU runtime recommended.
2. Run all cells from top to bottom.
3. MNIST is downloaded automatically through Keras; no manual dataset upload is required.
4. All generated plots are automatically saved to the `plots/` directory.
5. The notebook uses a fixed random seed of 42 for reproducibility.
6. The additional exercises can be run after completing the main sections.
