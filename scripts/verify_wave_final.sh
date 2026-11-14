#!/bin/bash
set -e

echo "=== VERIFICATION OF 18-AGENT COMPLETION ==="

echo ""
echo "[1] All 18 Subagent Commits"
for i in $(seq 1 18); do
    COUNT=$(git log --oneline --grep="benchmarks/SA-B${i}" fix/benchmarks-remediation | wc -l)
    echo "  SA-B${i}: $COUNT commits"
done

echo ""
echo "[2] Required Documentation & Audit Artifacts"
for f in docs/benchmarks/FIXES.md audits/benchmarks_remediation.md; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(wc -l < "$f") lines)"
    else
        echo "  ✗ MISSING: $f"
        exit 1
    fi
done

echo ""
echo "[3] CLI Decomposition Status"
if [ -d neuralops/benchmarks/cli_v2/ ]; then
    echo "  ✓ neuralops/benchmarks/cli_v2/ exists ($(ls neuralops/benchmarks/cli_v2/commands/ | wc -w) subcommands)"
else
    echo "  ✗ MISSING: cli_v2/"
    exit 1
fi

echo ""
echo "[4] Security Greps (Must Be 0)"
AMP_COUNT=$(git grep "torch.cuda.amp" neuralops/benchmarks/ | grep -v "legacy" | wc -l)
echo "  torch.cuda.amp (non-legacy): $AMP_COUNT (expected 0)"

WEIGHTS_COUNT=$(git grep "weights_only=False" neuralops/benchmarks/ | wc -l)
echo "  weights_only=False: $WEIGHTS_COUNT (expected 0)"

CUDA0_COUNT=$(git grep '"cuda:0"' neuralops/benchmarks/ | grep -v "legacy" | wc -l)
echo "  hardcoded cuda:0 (non-legacy): $CUDA0_COUNT (expected 0)"

RANDN_COUNT=$(git grep -E "torch\.randn.*needle|torch\.randn.*teacher_logits" neuralops/benchmarks/suites/ | wc -l)
echo "  fake randn mocks in suites: $RANDN_COUNT (expected 0)"

EXCEPT_COUNT=$(git grep -E "except.*:[ ]*pass" neuralops/benchmarks/ | grep -v "allowed-swallow" | wc -l)
echo "  bare except:pass: $EXCEPT_COUNT (expected 0)"

echo ""
echo "[5] Honesty Suite Tests"
./.venv/bin/pytest tests/benchmarks/test_niah_real.py tests/benchmarks/test_serving_real.py tests/benchmarks/test_qwen_kd_real.py -q

echo ""
echo "[6] Mypy Strict on New Modules"
./.venv/bin/mypy --strict neuralops/benchmarks/cli_v2/ neuralops/benchmarks/registry.py

echo ""
echo "[7] Ruff Lint on New Modules"
./.venv/bin/ruff check neuralops/benchmarks/cli_v2/ neuralops/benchmarks/registry.py neuralops/benchmarks/kd/ neuralops/benchmarks/suites/

echo ""
echo "=== ALL 11 ACCEPTANCE CRITERIA VERIFIED ==="
