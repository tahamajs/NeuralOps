#!/usr/bin/env bash
# Submit persistent, cluster-managed GPU jobs that run continuously and output to disk

set -euo pipefail

ssh -o ConnectTimeout=10 -o BatchMode=yes hpc "
    cat << 'PBSEOF' > ~/NeuralOps/run_screen_persistent.pbs
#!/usr/bin/env bash
#PBS -N neuralops_run
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONPATH=\"/home/tahamajlesi.ut/venv_cu113/lib/python3.10/site-packages:/home/tahamajlesi.ut/NeuralOps:\${PYTHONPATH:-}\"
export LD_LIBRARY_PATH=\"/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:/share/apps/gcc/gcc-8.2.0/lib64:/home/tahamajlesi.ut/venv_cu113/lib:\${LD_LIBRARY_PATH:-}\"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/tahamajlesi.ut/NeuralOps
mkdir -p results/screen_logs results/csv results/reports results/tables

LOG_FILE=/home/tahamajlesi.ut/NeuralOps/results/screen_logs/gpu_continuous_run.log
PYTHON_EXEC=/home/tahamajlesi.ut/.conda/envs/neuralops/bin/python

exec >> \"\$LOG_FILE\" 2>&1

echo \"=== STARTING PERSISTENT GPU PIPELINE: \$(date -u) ===\"
echo \"HOST: \$(hostname)\"
nvidia-smi

echo \"[*] Phase 1: High-Throughput Heavy GPU Stress Training & Distillation (3000 steps)...\"
\"\$PYTHON_EXEC\" -u neuralops/benchmarks/legacy_scripts/gpu_stress_heavy_trainer.py --steps 3000 --batch-size 8 --accum-steps 4

echo \"[*] Phase 2: Consolidate Master SQLite Ledger, Paper Tables, and Dashboard...\"
\"\$PYTHON_EXEC\" -u neuralops/benchmarks/legacy_scripts/consolidate_and_organize_benchmarks.py

echo \"=== GPU PIPELINE COMPLETED SUCCESSFULLY: \$(date -u) ===\"
PBSEOF

    JOB_ID=\$(qsub ~/NeuralOps/run_screen_persistent.pbs)
    echo \"[SUBMITTED] PBS job \${JOB_ID} queued!\"
"
