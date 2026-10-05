# 📊 experiment_6: Ultra-HD Deep JSCC (256×256)

## 🎯 Experiment Overview
- **Resolution**: 256×256 Ultra-HD patches from DIV2K
- **Architecture**: DeepJSCC-MS (Parameters: 608,225)
- **Loss**: COMPOSITE (Perceptual MSE + L1 + SSIM)
- **Best Validation PSNR**: **16.29 dB** @ 10.0 dB SNR

## 📈 Rate-Distortion Performance Across SNR
| Wireless Channel SNR (dB) | Reconstructed PSNR (dB) |
| :---: | :---: |
| -5.0 dB | **15.64 dB** |
| +0.0 dB | **16.20 dB** |
| +5.0 dB | **16.38 dB** |
| +10.0 dB | **16.43 dB** |
| +15.0 dB | **16.45 dB** |
| +20.0 dB | **16.45 dB** |
| +25.0 dB | **16.46 dB** |

## 🖼️ Visual Artifacts Generated
- `psnr_vs_snr.png`: Rate-Distortion performance across the wireless channel SNR spectrum.
- `reconstruction_grid.png`: High-resolution visual comparison at 0 dB, 10 dB, and 20 dB SNR.
- `training_curves.png`: Training & validation convergence curves.
