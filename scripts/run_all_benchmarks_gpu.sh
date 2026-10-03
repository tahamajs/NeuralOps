#!/bin/bash
# Master Shell Script to run ALL benchmarking suites on GPU with ONE command
# This automatically executes the Comprehensive Benchmark Orchestrator for all modules

echo "============================================================"
echo "LAUNCHING MASTER BENCHMARK ORCHESTRATOR ON GPU"
echo "============================================================"

# Ensure output directory exists based on new taxonomy
OUTPUT_DIR="results/07_misc_tests/master_benchmarks"
mkdir -p "$OUTPUT_DIR"

# Launch the master python orchestrator
# It automatically iterates over ALL registered modules (or those passed as arguments)
# and runs PPL, LM-Eval, and RULER sequentially.
python3 -m neuralops.benchmarks.run_all_benchmarks \
  --device cuda \
  --output_dir "$OUTPUT_DIR" \
  "$@"

echo "============================================================"
echo "ALL BENCHMARKS COMPLETED!"
echo "Master results aggregated at: $OUTPUT_DIR/aggregate_benchmarks.csv"
echo "============================================================"
