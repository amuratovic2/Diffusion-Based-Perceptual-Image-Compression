$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Missing .venv. Run .\scripts\setup_windows.ps1 first."
    exit 1
}

Set-Location $ProjectRoot

& $Python src\prepare_subset.py `
  --input external\CDC_compression\imgs `
  --output data\smoke_cdc_imgs `
  --limit 3 `
  --size 256

& $Python src\baseline_compress.py `
  --input data\smoke_cdc_imgs `
  --output results\smoke_jpeg_q35 `
  --codec jpeg `
  --quality 35

& $Python src\baseline_compress.py `
  --input data\smoke_cdc_imgs `
  --output results\smoke_webp_q35 `
  --codec webp `
  --quality 35

& $Python src\evaluate_metrics.py `
  --ref data\smoke_cdc_imgs `
  --recon results\smoke_jpeg_q35\images `
  --bitrate-csv results\smoke_jpeg_q35\bitrates.csv `
  --output results\smoke_jpeg_q35\metrics.csv

& $Python src\evaluate_metrics.py `
  --ref data\smoke_cdc_imgs `
  --recon results\smoke_webp_q35\images `
  --bitrate-csv results\smoke_webp_q35\bitrates.csv `
  --output results\smoke_webp_q35\metrics.csv

Write-Host "Smoke test finished. See results\smoke_*."

