#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# Execute all active examples and record logs and timings.

set -uo pipefail

mkdir -p audits/wave6/logs
REPORT="audits/wave6/all_examples_run.md"

echo "# NeuralOps Wave 6 — Full Examples Execution Log" > "$REPORT"
echo "" >> "$REPORT"
echo "| Example File | Exit Code | Runtime | Status |" >> "$REPORT"
echo "|---|:---:|:---:|:---:|" >> "$REPORT"

TOTAL=0
PASSED=0
FAILED=0

for f in examples/[0-9]*.py; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    log_file="audits/wave6/logs/${base%.py}.log"
    TOTAL=$((TOTAL + 1))
    
    echo -n "Running $base... "
    t0=$(date +%s)
    if python "$f" > "$log_file" 2>&1; then
        code=0
        status="PASS"
        PASSED=$((PASSED + 1))
        echo "OK"
    else
        code=$?
        status="FAIL"
        FAILED=$((FAILED + 1))
        echo "FAIL (exit $code)"
    fi
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    
    echo "| \`$base\` | $code | ${elapsed}s | **$status** |" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "Total: $TOTAL, Passed: $PASSED, Failed: $FAILED" >> "$REPORT"
echo "Execution finished: $PASSED/$TOTAL passed."
