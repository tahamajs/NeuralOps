#!/usr/bin/env bash
set -euo pipefail

echo "=== NeuralOps v2.4.0 Release Verification ==="
echo "Running at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Git commit: $(git rev-parse HEAD)"
echo ""

echo "--- Step 1: Version check ---"
python -m neuralops --version
echo ""

echo "--- Step 2: Unit and contract tests ---"
python -m pytest tests/unit tests/contract -q -m "not hub and not gpu"
echo ""

echo "--- Step 3: Ruff linting ---"
python -m ruff check neuralops/ tests/
echo ""

echo "--- Step 4: CHANGELOG version presence ---"
grep -c "2.4.0" CHANGELOG.md
echo ""

echo "--- Step 5: Critical file presence check ---"
for f in CHANGELOG.md BREAKING_CHANGES.md docs/truth_audit.md issues/wave6_hotfix_backlog.md; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        echo "PRESENT: $f"
    else
        echo "MISSING OR EMPTY: $f" >&2
        exit 1
    fi
done
echo ""

echo "=== ALL GATES PASSED ==="
