#!/bin/bash
# ==============================================================
# NeuralOps GPU verification job — Sharif HPC Cluster
# Canonical template. Copy to ~/NeuralOps/run.sh and adjust
# `#PBS -q` to match current queue availability (qstat -Q).
# ==============================================================

#PBS -N neuralops_gpu_verify
#PBS -m abe
#PBS -M tahamajlesi@ut.ac.ir
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=16G
#PBS -q cuda7

# --- Environment ---
# Adjust conda path if needed (anaconda3.8, anaconda3.11)
source /share/apps/Anaconda/anaconda3.8/bin/activate neuralops

cd $PBS_O_WORKDIR

# --- Diagnostics ---
echo "=== Job metadata ==="
hostname
date
echo "PBS_JOBID=$PBS_JOBID"
echo "PBS_O_WORKDIR=$PBS_O_WORKDIR"

echo "=== Python ==="
which python
python --version

echo "=== CUDA / torch ==="
python - <<'PYEOF'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PYEOF

echo "=== Triton ==="
python -c "import triton; print(triton.__version__)" 2>&1 || echo "triton not importable"

# --- Test suites ---
echo "=== Triton parity tests ==="
python -m pytest tests/triton_parity/ -v --tb=short
TRITON_EXIT=$?

echo "=== E2E DoD tests ==="
python -m pytest tests/e2e/test_v01_dod_distilbert.py -v --tb=short
E2E_EXIT=$?

echo "=== L4 empirical gates ==="
python -m pytest tests/validation/test_l4_empirical_gates.py -v --tb=short
L4_EXIT=$?

echo "=== Summary ==="
echo "triton_parity exit: $TRITON_EXIT"
echo "e2e_dod exit: $E2E_EXIT"
echo "l4_empirical exit: $L4_EXIT"
date

# Exit with non-zero if any test failed
if [ $TRITON_EXIT -ne 0 ] || [ $E2E_EXIT -ne 0 ] || [ $L4_EXIT -ne 0 ]; then
  exit 1
fi
exit 0
