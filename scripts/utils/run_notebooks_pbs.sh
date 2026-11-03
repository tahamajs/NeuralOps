#!/bin/bash
#PBS -N neuralops_29nb
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -V
#PBS -o /home/tahamajlesi.ut/NeuralOps/results/notebooks_pbs.out
#PBS -e /home/tahamajlesi.ut/NeuralOps/results/notebooks_pbs.err

export PATH="/home/tahamajlesi.ut/bin:/share/apps/gcc/gcc-8.2.0/bin:$PATH"
export LD_LIBRARY_PATH="/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-B/home/tahamajlesi.ut/bin"
export COMPILER_PATH="/home/tahamajlesi.ut/bin"
export PYTHONPATH="/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd /home/tahamajlesi.ut/NeuralOps
mkdir -p results

echo "======================================================================"
echo "Starting NeuralOps 29 Notebook Suite on $(hostname) at $(date)"
echo "Target Accelerator:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "======================================================================"

/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python -u \
    scripts/utils/run_all_notebooks.py \
    --dir notebooks \
    --summary-file results/NOTEBOOKS_EXECUTION_SUMMARY.md

EXIT_CODE=$?
echo "======================================================================"
echo "Finished NeuralOps 29 Notebook Suite at $(date) with exit code ${EXIT_CODE}"
echo "======================================================================"
exit ${EXIT_CODE}
