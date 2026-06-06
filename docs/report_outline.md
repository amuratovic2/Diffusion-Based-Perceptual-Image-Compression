# Report Outline

Use the IEEE conference template and keep the final report under 10 pages.

## Title

Adapting Diffusion-Based Perceptual Image Compression to ImageNet-1k

## 1. Introduction

- Explain lossy image compression.
- Explain the distortion/perception tradeoff.
- State that CorrDiff is the assigned paper.
- State that official CorrDiff code was unavailable, so CDC was used as a
  reproducible diffusion-based compression proxy.

## 2. Assigned Paper: CorrDiff

- Problem: diffusion reconstructions look realistic but may drift from the
  source image.
- Main idea: privileged end-to-end decoder corrects diffusion sampling.
- Components: encoder, entropy model, correction decoder, score network.
- Expected benefit: better perception while controlling distortion.

## 3. Implemented Method: CDC

- Conditional diffusion codec.
- Compression model produces conditioning/context.
- Diffusion decoder reconstructs perceptual image from compressed context.
- x-parameterization and epsilon-parameterization variants.

## 4. Implementation

- Upstream code: `external/CDC_compression`.
- Added scripts:
  - ImageNet subset preparation.
  - JPEG/WebP baselines.
  - CDC folder inference wrapper.
  - Metric evaluation.
- Dataset: ImageNet-1k validation subset, resized to 256x256.

## 5. Metrics

- bpp: compressed bits per pixel, lower means stronger compression.
- PSNR: pixel fidelity, higher is better.
- SSIM/MS-SSIM: structural similarity, higher is better.
- LPIPS: learned perceptual distance, lower is better.
- FID: distribution-level realism, lower is better; optional if enough samples
  and feature extractor are available.

## 6. Results

Insert a table:

| Method | Quality | bpp | PSNR | SSIM | LPIPS | MS-SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JPEG | 35 | | | | | |
| WebP | 35 | | | | | |
| CDC xparam | 65 steps | | | | | |

Add visual examples:

- Original.
- JPEG/WebP reconstruction.
- CDC reconstruction.

## 7. Problems Faced

- CorrDiff official repository had no released implementation.
- Local machine initially had Python 3.14 only, while PyTorch projects target
  Python 3.8-3.10.
- No ImageNet data found locally during setup.
- No GPU detected, so CDC inference may be slow on CPU.

## 8. Possible Improvements

- Run on the full ImageNet validation set.
- Use a CUDA GPU and larger CDC sample counts.
- Add FID with thousands of samples.
- Compare against CompressAI learned codecs.
- Re-run with CorrDiff if official code/checkpoints are released.

