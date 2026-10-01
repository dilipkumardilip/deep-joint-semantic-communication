# Experiment 1: Deep JSCC Baseline Verification

## 1. Overview & Objective
Establish a functional baseline for the Deep Joint Source-Channel Communication (Deep JSCC) pipeline based on the Bourtsoulatze et al. (2019) architecture for wireless image transmission over an AWGN channel.

- **Date:** October 2, 2026
- **Dataset:** CIFAR-10 ($32 \times 32 \times 3$)
- **Hardware:** Apple Silicon GPU (`mps`)
- **Status:** Completed (Baseline Verified)

---

## 2. Configuration & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | DeepJSCC (5 Convs + 5 Deconvs) | Kernel size $5\times 5$, PReLU activations |
| **Channel Bandwidth $c$** | 16 | Latent feature channels ($k = 16 \times 8 \times 8 = 1024$ channel symbols) |
| **Bandwidth Ratio $k/n$** | $1024 / 3072 = 1/3$ | Compression ratio ($3\times$ compression) |
| **Transmit Power $P$** | $1.0$ | Average symbol power constraint $\frac{1}{k}\sum z_i^2 = 1$ |
| **Training Split** | 70% (35,000 images) | Random split |
| **Validation Split** | 30% (15,000 images) | Held-out validation |
| **Test Set** | 10,000 images | Standard CIFAR-10 test set |
| **Epochs** | 1 (Sanity / Baseline) | Quick initial run |
| **Batch Size** | 128 | |
| **Optimizer** | Adam | Initial $\text{LR} = 10^{-3}$ |
| **LR Scheduler** | None (Static LR) | Flat learning rate of $0.001$ |
| **Channel Condition** | AWGN @ $10.0$ dB | Nominal training SNR |

---

## 3. Results & Convergence

### Epoch Progression
| Epoch | Learning Rate | Train MSE | Train PSNR (dB) | Val MSE | Val PSNR (dB) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.00100 | 0.02501 | 16.02 | 0.01533 | 18.15 | * Best Checkpoint |

### Held-out Test Set Performance (10,000 Images)
- **Final Test MSE:** `0.01529`
- **Final Test PSNR:** `18.16 dB`

### Robustness across Channel SNRs (Deep JSCC Characteristic Curve)
| Channel SNR (dB) | Evaluated MSE | Reconstruction PSNR (dB) |
| :---: | :---: | :---: |
| **0.0 dB** | 0.01875 | 17.27 |
| **5.0 dB** | 0.01613 | 17.92 |
| **10.0 dB** | 0.01533 | 18.15 |
| **15.0 dB** | 0.01508 | 18.22 |
| **20.0 dB** | 0.01500 | 18.24 |

---

## 4. Observations & Findings

1. **Power Normalization Integrity:**
   - The power normalization layer strictly constrained the transmitted symbol energy to $1.0000$, validating the physical wireless transmission assumption.

2. **Graceful Degradation:**
   - Even when channel SNR dropped by $20$ dB (from $20$ dB down to $0$ dB), PSNR only dropped by $\sim 0.97$ dB. The system demonstrated strong resilience against channel degradation without catastrophic "cliff-effect" failure.

3. **Bottlenecks Identified:**
   - With only 1 epoch and static learning rate, the network reconstructed general object shapes and color silhouettes, but fine textures and high-frequency edges remained blurry.
   - **Recommendation for Experiment 2:** Increase epochs to 50 and implement a decaying learning rate scheduler (Cosine Annealing) to allow deep fine-tuning of the convolutional kernels.
