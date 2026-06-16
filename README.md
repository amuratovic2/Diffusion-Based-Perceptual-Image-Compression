# Diffusion-Based Perceptual Image Compression on ImageNet-1k

This repository contains the code used for a course project on diffusion-based
perceptual image compression.

The assigned paper was:

> Correcting Diffusion-Based Perceptual Image Compression with Privileged
> End-to-End Decoder, ICML 2024.

The official CorrDiff repository did not provide runnable training/evaluation
code or checkpoints during the project, so the implementation uses a related
open-source method:

> Lossy Image Compression with Conditional Diffusion Models (CDC), NeurIPS 2023.

The original CDC code is kept under `external/CDC_compression`. Project-specific
adaptation scripts are kept under `src/`.

## Repository Contents

```text
src/
  prepare_subset.py       Prepare a resized ImageNet subset.
  baseline_compress.py    Run JPEG/WebP baselines.
  run_cdc.py              Run CDC inference on a folder of images.
  evaluate_metrics.py     Compute bpp, PSNR, SSIM, and MAE.
  summarize_results.py    Aggregate metric CSV files.

scripts/
  setup_macos.sh
  setup_windows.ps1
  run_smoke_test_macos.sh
  run_smoke_test.ps1
  download_cdc_weights.md

external/CDC_compression/
  Upstream CDC implementation used by src/run_cdc.py.
```

Large/generated files are intentionally not tracked:

- virtual environments
- ImageNet data
- experiment results
- model checkpoints
- generated documentation/presentations

## Setup

Create the Python environment:

```bash
chmod +x scripts/setup_macos.sh
./scripts/setup_macos.sh
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

## Checkpoint

Download the CDC pretrained weights manually from:

```text
https://huggingface.co/rhyang/CDC_params
```

Place the x-parameterization checkpoint here:

```text
checkpoints/cdc_xparam_b00032_aux0.pt
```

The checkpoint used in the project corresponds to the released CDC
x-parameterization model:

```text
x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt
```

## Data

Place or extract the ImageNet-1k validation dataset locally. The expected folder
shape is:

```text
imagenet-val/
  n01440764/
    ILSVRC2012_val_00000293.JPEG
  n01443537/
    ILSVRC2012_val_00000236.JPEG
  ...
```

Prepare a reproducible random subset:

```bash
./.venv/bin/python src/prepare_subset.py \
  --input /path/to/imagenet-val \
  --output data/imagenet_subset_100_random_256 \
  --limit 100 \
  --size 256 \
  --shuffle \
  --seed 42
```

## Baselines

Run JPEG and WebP baselines:

```bash
./.venv/bin/python src/baseline_compress.py \
  --input data/imagenet_subset_100_random_256 \
  --output results/imagenet100_random_jpeg_q35 \
  --codec jpeg \
  --quality 35

./.venv/bin/python src/baseline_compress.py \
  --input data/imagenet_subset_100_random_256 \
  --output results/imagenet100_random_webp_q35 \
  --codec webp \
  --quality 35
```

## CDC Inference

Run CDC with the x-parameterization checkpoint:

```bash
./.venv/bin/python src/run_cdc.py \
  --variant xparam \
  --ckpt checkpoints/cdc_xparam_b00032_aux0.pt \
  --input data/imagenet_subset_100_random_256 \
  --output results/imagenet100_random_cdc_xparam_steps20 \
  --lpips-weight 0.0 \
  --steps 20 \
  --device auto
```

## Metrics

Evaluate reconstructions:

```bash
./.venv/bin/python src/evaluate_metrics.py \
  --ref data/imagenet_subset_100_random_256 \
  --recon results/imagenet100_random_jpeg_q35/images \
  --bitrate-csv results/imagenet100_random_jpeg_q35/bitrates.csv \
  --output results/imagenet100_random_jpeg_q35/metrics.csv

./.venv/bin/python src/evaluate_metrics.py \
  --ref data/imagenet_subset_100_random_256 \
  --recon results/imagenet100_random_webp_q35/images \
  --bitrate-csv results/imagenet100_random_webp_q35/bitrates.csv \
  --output results/imagenet100_random_webp_q35/metrics.csv

./.venv/bin/python src/evaluate_metrics.py \
  --ref data/imagenet_subset_100_random_256 \
  --recon results/imagenet100_random_cdc_xparam_steps20/images \
  --bitrate-csv results/imagenet100_random_cdc_xparam_steps20/bitrates.csv \
  --output results/imagenet100_random_cdc_xparam_steps20/metrics.csv
```

Summarize results:

```bash
./.venv/bin/python src/summarize_results.py \
  --inputs \
    results/imagenet100_random_jpeg_q35/metrics.csv \
    results/imagenet100_random_webp_q35/metrics.csv \
    results/imagenet100_random_cdc_xparam_steps20/metrics.csv \
  --output results/imagenet100_random_summary.csv
```

Implemented metrics:

- bpp
- PSNR
- SSIM
- MAE

## Notes

This repository does not include ImageNet images or model checkpoints. They must
be downloaded separately.

This is not a full CorrDiff reproduction. It is a reproducible CDC-based
implementation and evaluation pipeline for the same research area.
