#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x ./.venv/bin/python ]; then
  echo "Missing .venv. Run ./scripts/setup_macos.sh first."
  exit 1
fi

./.venv/bin/python src/prepare_subset.py \
  --input external/CDC_compression/imgs \
  --output data/smoke_cdc_imgs \
  --limit 3 \
  --size 256

./.venv/bin/python src/baseline_compress.py \
  --input data/smoke_cdc_imgs \
  --output results/smoke_jpeg_q35 \
  --codec jpeg \
  --quality 35

./.venv/bin/python src/baseline_compress.py \
  --input data/smoke_cdc_imgs \
  --output results/smoke_webp_q35 \
  --codec webp \
  --quality 35

./.venv/bin/python src/evaluate_metrics.py \
  --ref data/smoke_cdc_imgs \
  --recon results/smoke_jpeg_q35/images \
  --bitrate-csv results/smoke_jpeg_q35/bitrates.csv \
  --output results/smoke_jpeg_q35/metrics.csv

./.venv/bin/python src/evaluate_metrics.py \
  --ref data/smoke_cdc_imgs \
  --recon results/smoke_webp_q35/images \
  --bitrate-csv results/smoke_webp_q35/bitrates.csv \
  --output results/smoke_webp_q35/metrics.csv

echo "Smoke test finished. See results/smoke_*."
