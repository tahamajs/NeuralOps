#!/usr/bin/env bash
# ==============================================================================
# NeuralOps Direct PyPI Release Script (v1.4.3)
# ==============================================================================
set -euo pipefail

echo "======================================================================"
echo "  🚀 NEURALOPS PYPI DIRECT RELEASE SCRIPT (v1.4.3)"
echo "======================================================================"

PYTHON_BIN="/mnt/d/venv/bin/python3"
TWINE_BIN="/mnt/d/venv/bin/twine"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "[-] Virtual environment not found at $PYTHON_BIN. Falling back to python3..."
    PYTHON_BIN="python3"
    TWINE_BIN="twine"
fi

echo "[2/5] Cleaning existing build and distribution directories..."
rm -rf dist/ build/ *.egg-info

echo "[3/5] Building clean sdist and wheel packages..."
$PYTHON_BIN -m build --no-isolation

echo "[4/5] Validating distribution packages with twine check..."
$TWINE_BIN check dist/*

echo "[5/5] Checking PyPI credentials and uploading artifacts..."
if [ -z "${TWINE_PASSWORD:-}" ] && [ ! -f "$HOME/.pypirc" ]; then
    echo "[-] TWINE_PASSWORD environment variable or ~/.pypirc is required to upload."
    echo "[!] Run with TWINE_PASSWORD=pypi-... $0 to proceed."
    exit 1
fi

$TWINE_BIN upload --skip-existing dist/*

echo "======================================================================"
echo "  ✅ NEURALOPS v1.4.3 SUCCESSFULLY UPLOADED TO PYPI!"
echo "======================================================================"
