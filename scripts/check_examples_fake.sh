#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# Anti-fake verification linter for NeuralOps examples suite.
# Checks:
# 1. No hardcoded pass_fail literal True
# 2. No silent except ImportError: pass
# 3. No inline def pillar... reimplementations
# 4. No unredacted HuggingFace tokens

set -euo pipefail

echo "========================================================"
echo "  NeuralOps Anti-Fake Examples Static Linter"
echo "========================================================"

VIOLATIONS=0

echo "[1/4] Checking for hardcoded pass_fail True..."
if grep -rn --include="*.py" 'pass_fail("Overall.*True' examples/; then
    echo "  [ERROR] Hardcoded pass_fail True found in examples!"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "  [OK] No hardcoded pass_fail True."
fi

echo "[2/4] Checking for silent except ImportError: pass..."
if grep -rn -A 1 --include="*.py" 'except ImportError' examples/ | grep -B 1 'pass'; then
    echo "  [ERROR] Silent except ImportError: pass found in examples!"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "  [OK] No silent except ImportError: pass."
fi

echo "[3/4] Checking for inline pillar implementations..."
if grep -rn --include="*.py" '^def pillar' examples/; then
    echo "  [ERROR] Inline def pillar... found in examples!"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "  [OK] No inline def pillar... implementations."
fi

echo "[4/4] Checking for unredacted HuggingFace tokens..."
if grep -rnE --include="*.py" 'hf_[a-zA-Z0-9]{34}' examples/; then
    echo "  [ERROR] Unredacted HuggingFace token found in examples!"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo "  [OK] No unredacted HF tokens."
fi

echo "--------------------------------------------------------"
if [ "$VIOLATIONS" -gt 0 ]; then
    echo "FAILED: $VIOLATIONS violation(s) detected across examples/."
    exit 1
fi
echo "SUCCESS: All anti-fake static checks passed cleanly."
exit 0
