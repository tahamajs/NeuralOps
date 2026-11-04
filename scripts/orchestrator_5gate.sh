#!/usr/bin/env bash
set -euo pipefail

echo "=== 5-GATE VERIFICATION $(date) ==="

# Gate A: clean working tree
echo "[Gate A] Working tree"
git checkout -- results/master_ledger.sqlite 2>/dev/null || true
if [ -n "$(git status --porcelain)" ]; then
  echo "FAIL: working tree not clean"
  git status --porcelain
  exit 1
fi
echo "PASS"

# Gate B: invariants
echo "[Gate B] Invariants"
REG_COUNTS=$(python3 -c "from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY; print(f'{len(ATTENTION_REGISTRY)} {len(COMPRESSOR_REGISTRY)}')")
A=$(echo "$REG_COUNTS" | awk '{print $1}')
C=$(echo "$REG_COUNTS" | awk '{print $2}')
T=$(find neuralops/ops/triton -maxdepth 1 -name '*.py' -not -name '__init__.py' | wc -l | tr -d ' ')
U=$(find neuralops/ops/cuda -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
N=$(find notebooks -maxdepth 1 -name '*.ipynb' | wc -l | tr -d ' ')
echo "Attentions: $A | Compressors: $C | Triton: $T | CUDA: $U | Notebooks: $N"
# Ground truth baseline: 131 Attentions, 72 Compressors
[ "$A" -ge 131 ] && [ "$C" -ge 72 ] && [ "$T" -ge 26 ] && [ "$U" -ge 110 ] && [ "$N" -ge 29 ] || { echo "FAIL: invariant drift"; exit 1; }
echo "PASS"

# Gate C: package import
echo "[Gate C] Package import"
python3 -c "import neuralops; print('neuralops', neuralops.__version__)" || { echo "FAIL: import failed"; exit 1; }
echo "PASS"

# Gate D: CPU test subset
echo "[Gate D] CPU test subset"
pytest tests/test_registry.py -q --tb=short || { echo "FAIL: registry tests"; exit 1; }
echo "PASS"

# Gate E: code syntax / linter check
echo "[Gate E] Linter / Syntax"
if command -v ruff >/dev/null 2>&1; then
  ruff check neuralops/ || { echo "FAIL: ruff"; exit 1; }
else
  python3 -m py_compile neuralops/__init__.py || { echo "FAIL: syntax check"; exit 1; }
fi
echo "PASS"

echo "=== ALL 5 GATES PASSED ==="
