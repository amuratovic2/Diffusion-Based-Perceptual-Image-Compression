#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing python3. Install it with Homebrew:"
  echo "  brew install python@3.10"
  exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PYTHON_VERSION" in
  3.10|3.11|3.12)
    ;;
  *)
    echo "Detected Python $PYTHON_VERSION. Prefer Python 3.10, 3.11, or 3.12 for PyTorch on Apple Silicon."
    echo "With Homebrew:"
    echo "  brew install python@3.10"
    echo "  PYTHON_BIN=/opt/homebrew/bin/python3.10 ./scripts/setup_macos.sh"
    exit 1
    ;;
esac

"$PYTHON_BIN" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
PY

echo ""
echo "Environment ready."
echo "Run:"
echo "  ./scripts/run_smoke_test_macos.sh"

