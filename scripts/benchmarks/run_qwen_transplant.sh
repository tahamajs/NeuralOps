#!/bin/bash
#PBS -N qwen_transplant
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o /home/tahamajlesi.ut/NeuralOps/qwen_transplant.log

cd /home/tahamajlesi.ut/NeuralOps
export PYTHONUNBUFFERED=1
export PATH=/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin:/home/tahamajlesi.ut/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH

echo "=== PBS Job Started at $(date) ==="
nvidia-smi

/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python -u scripts/benchmarks/qwen_7b_transplant.py

echo "=== PBS Job Finished at $(date) ==="
nvidia-smi
