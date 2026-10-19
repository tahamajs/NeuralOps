#!/usr/bin/env bash
set -euo pipefail

# Reproducibility probe: all runtime inspection occurs on the host that runs
# this script; no repository-specific absolute path is required.
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"
sha="$(git rev-parse HEAD)"
report="docs/reports/REPRO_${sha}.md"
mkdir -p "$(dirname "$report")"

python_bin="${PYTHON_BIN:-python3}"
{
  echo "# NeuralOps Reproducibility Report"
  echo
  echo "- Commit: $sha"
  echo "- Python: $($python_bin --version 2>&1)"
  "$python_bin" - <<'PY'
import importlib.util
import platform
import subprocess

print(f"- Platform: {platform.platform()}")
for name in ("torch", "triton"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"- {name}: not installed")
        continue
    module = __import__(name)
    print(f"- {name}: {getattr(module, '__version__', 'unknown')}")
try:
    import torch
    print(f"- CUDA: {torch.version.cuda or 'unavailable'}")
    print(f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
except Exception as exc:
    print(f"- PyTorch runtime: unavailable ({exc})")
PY
  echo
  echo "## Registry counts"
  if "$python_bin" -c 'import neuralops' >/dev/null 2>&1; then
    "$python_bin" - <<'PY'
from neuralops import list_attentions, list_compressors, list_triton_kernels, list_cuda_kernels
print(f"- Attentions: {len(list_attentions())}")
print(f"- Compressors: {len(list_compressors())}")
print(f"- Triton operators: {len(list_triton_kernels())}")
print(f"- CUDA catalog entries: {len(list_cuda_kernels())}")
PY
  else
    echo "- Registry counts: unavailable"
  fi
  echo
  echo "## Fast verification"
  if "$python_bin" -c 'import pytest' >/dev/null 2>&1; then
    "$python_bin" -m pytest -q tests/test_fast.py
  else
    echo "pytest: unavailable"
  fi
} > "$report"
echo "$report"
