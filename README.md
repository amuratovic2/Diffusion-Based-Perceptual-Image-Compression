# Correcting Diffusion-Based Perceptual Image Compression - Project Workspace

This project reproduces an experiment in the same research area as the assigned
CorrDiff paper:

> Correcting Diffusion-Based Perceptual Image Compression with Privileged
> End-to-End Decoder, ICML 2024.

The official CorrDiff repository currently contains only a placeholder README,
so this workspace uses the closest practical open-source alternative:

> Lossy Image Compression with Conditional Diffusion Models (CDC), NeurIPS 2023.

CDC is a diffusion-based perceptual image compression method and is one of the
directly relevant methods for discussing CorrDiff. The original CDC code is kept
unchanged under `external/CDC_compression`; all ImageNet adaptation, baselines,
metrics, and experiment orchestration live in `src/` and `scripts/`.

## Current Machine Notes

The machine check found initially:

- Git is installed.
- Python 3.14 was installed globally.
- Python 3.10 has now been installed for this project.
- A local `.venv` has been created with PyTorch CPU, torchvision, LPIPS,
  MS-SSIM, EMA PyTorch, scikit-image, and CDC dependencies.
- CUDA is not available in this environment, so inference is CPU-only.
- CompressAI is optional and may require Microsoft C++ Build Tools on Windows.
- Conda/Mamba are not installed.
- `nvidia-smi` is not available in PATH, so GPU support is not assumed.

Use the setup scripts below to create a Python 3.10 virtual environment for this
project. PyTorch support for Python 3.14 is not a good target for this codebase.

## Recommended Workflow

### macOS on Apple Silicon

For an M1/M2/M3/M4 Mac, use the macOS setup script. PyTorch will use Apple GPU
acceleration through the `mps` backend when available.

```bash
chmod +x scripts/setup_macos.sh scripts/run_smoke_test_macos.sh
./scripts/setup_macos.sh
./scripts/run_smoke_test_macos.sh
```

Check that MPS is available:

```bash
./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Run CDC on Apple GPU:

```bash
./.venv/bin/python src/run_cdc.py \
  --variant xparam \
  --ckpt checkpoints/cdc_xparam_b00032_aux0.pt \
  --input data/smoke_cdc_imgs \
  --output results/cdc_xparam_smoke_mps \
  --lpips-weight 0.0 \
  --steps 65 \
  --device auto
```

If `auto` detects MPS, it will use it. You can force it with `--device mps`.

### Windows CPU

### 1. Create Environment

Run PowerShell from this folder:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

If Python 3.10 is not installed, the script prints the exact `winget` command to
install it. After installing Python 3.10, run the setup script again.

### 2. Smoke Test On CDC Sample Images

This does not need ImageNet. It uses the three Kodak sample images bundled with
CDC and creates JPEG/WebP baselines plus metrics:

```powershell
.\scripts\run_smoke_test.ps1
```

### 3. Prepare An ImageNet Subset

Use your ImageNet validation folder, typically shaped as:

```text
ImageNet/val/n01440764/ILSVRC2012_val_00000293.JPEG
ImageNet/val/n01443537/ILSVRC2012_val_00000236.JPEG
...
```

Prepare a flat resized subset:

```powershell
.\.venv\Scripts\python.exe src\prepare_subset.py `
  --input "D:\datasets\imagenet\val" `
  --output data\imagenet_subset_256 `
  --limit 1000 `
  --size 256
```

### 4. Run Baselines

```powershell
.\.venv\Scripts\python.exe src\baseline_compress.py `
  --input data\imagenet_subset_256 `
  --output results\baseline_jpeg_q35 `
  --codec jpeg `
  --quality 35

.\.venv\Scripts\python.exe src\baseline_compress.py `
  --input data\imagenet_subset_256 `
  --output results\baseline_webp_q35 `
  --codec webp `
  --quality 35
```

### 5. Run CDC

Download model weights from the CDC Hugging Face page:

https://huggingface.co/rhyang/CDC_params

The smoke-tested checkpoint is:

```text
checkpoints\cdc_xparam_b00032_aux0.pt
```

It corresponds to `x_param/image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt`.

Then run either x-parameterization or epsilon-parameterization. Start with
x-parameterization because its default denoising step count is lower.

```powershell
.\.venv\Scripts\python.exe src\run_cdc.py `
  --variant xparam `
  --ckpt "checkpoints\cdc_xparam_b00032_aux0.pt" `
  --input data\imagenet_subset_256 `
  --output results\cdc_xparam `
  --lpips-weight 0.0 `
  --steps 65 `
  --device auto
```

If your checkpoint was trained with LPIPS loss, use `--lpips-weight 0.9`.

For a quick CPU smoke test, use fewer denoising steps:

```powershell
.\.venv\Scripts\python.exe src\run_cdc.py `
  --variant xparam `
  --ckpt checkpoints\cdc_xparam_b00032_aux0.pt `
  --input data\smoke_cdc_imgs `
  --output results\cdc_xparam_smoke_steps1 `
  --lpips-weight 0.0 `
  --steps 1 `
  --device auto
```

### 6. Evaluate Metrics

```powershell
.\.venv\Scripts\python.exe src\evaluate_metrics.py `
  --ref data\imagenet_subset_256 `
  --recon results\baseline_jpeg_q35\images `
  --bitrate-csv results\baseline_jpeg_q35\bitrates.csv `
  --output results\baseline_jpeg_q35\metrics.csv

.\.venv\Scripts\python.exe src\evaluate_metrics.py `
  --ref data\imagenet_subset_256 `
  --recon results\cdc_xparam\images `
  --bitrate-csv results\cdc_xparam\bitrates.csv `
  --output results\cdc_xparam\metrics.csv
```

Aggregate several metric CSV files into one summary table:

```powershell
.\.venv\Scripts\python.exe src\summarize_results.py `
  --inputs results\baseline_jpeg_q35\metrics.csv results\baseline_webp_q35\metrics.csv results\cdc_xparam\metrics.csv `
  --output results\summary.csv
```

Metrics implemented directly: bpp, PSNR, SSIM, MAE.

Optional metrics, if packages are installed: LPIPS and MS-SSIM. FID is left as a
documented optional extension because it needs a heavier feature extractor and
larger image batches to be meaningful.

CompressAI is not required for the CDC experiment. If you want additional
learned-codec baselines, install `requirements-optional.txt` after installing
Microsoft C++ Build Tools.

## Suggested Report Angle

In the report, state clearly:

- CorrDiff is the assigned paper.
- Its official code was not available at implementation time.
- CDC was selected as a reproducible diffusion-based perceptual image
  compression alternative.
- The project adapts CDC-style inference and metric evaluation to ImageNet-1k.
- The limitation is that this is not a full CorrDiff reproduction.
