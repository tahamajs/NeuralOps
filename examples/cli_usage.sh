#!/bin/bash
# Example: Using the NeuralOps Unified Command-Line Interface (CLI)
#
# Shows how to:
#   1. Verify system environment, PyTorch installation, and GPU/CPU devices.
#   2. Plan execution grids for specific module families (--all-gated, --all-delta).
#   3. Run knowledge distillation sweeps across subquadratic attention modules.

set -e

echo "=== 1. System Environment Verification ==="
python -m neuralops.benchmarks.cli verify

echo "=== 2. Plan Grid for Gated Attention Family ==="
python -m neuralops.benchmarks.cli plan --all-gated

echo "=== 3. Plan Grid for Delta Rule Family ==="
python -m neuralops.benchmarks.cli plan --all-delta

echo "=== 4. Example Run Command (Dry-run / Help) ==="
python -m neuralops.benchmarks.cli run --help | head -25

echo "CLI usage example completed successfully."
