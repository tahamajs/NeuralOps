#!/bin/bash
set -e

echo "=== WAVE 5 FINAL ACCEPTANCE ==="

# 1. Imports
for m in neuralops neuralops.layers neuralops.compression \
         neuralops.surgery neuralops.hub neuralops.validation \
         neuralops.serving neuralops.tracking neuralops.runtime \
         neuralops.distributed neuralops.trainer neuralops.cli; do
    ./.venv/bin/python -c "import $m" || { echo "FAIL: import $m"; exit 1; }
done
echo "✓ imports"

# 2. Tests
./.venv/bin/pytest tests/unit tests/contract -x -q || { echo "FAIL: unit/contract"; exit 1; }
./.venv/bin/pytest tests/integration -x -q -m "not hub and not gpu" || { echo "FAIL: integration"; exit 1; }
echo "✓ tests"

# 3. Lint
./.venv/bin/ruff check neuralops/ tests/ || { echo "FAIL: ruff"; exit 1; }
echo "✓ ruff"

# 4. Security
if grep -rn "torch.cuda.amp" neuralops/ | grep -v "test_" | grep -v ".pyc"; then
    echo "FAIL: deprecated torch.cuda.amp"; exit 1
fi
if grep -rn "weights_only=False" neuralops/ | grep -v "# noqa" | grep -v ".pyc"; then
    echo "FAIL: unsafe weights_only=False"; exit 1
fi
echo "✓ security greps"

# 5. Version
VERSION=$(./.venv/bin/python -c "from neuralops._version import __version__; print(__version__)")
echo "✓ current version is $VERSION"

# 6. Release artifacts
for f in CHANGELOG.md BREAKING_CHANGES.md AUDIT_MASTER_FIXES.md VERIFICATION.md; do
    [ -f "$f" ] || { echo "FAIL: missing $f"; exit 1; }
done
echo "✓ release artifacts"

# 7. Reports
[ -f audits/wave5_master.md ] || { echo "FAIL: missing wave5_master.md"; exit 1; }
echo "✓ wave5_master.md"

echo ""
echo "=== ALL CHECKS PASSED — READY TO TAG v2.4.0 ==="
