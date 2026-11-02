#!/bin/bash
#PBS -N triton_diag
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /home/tahamajlesi.ut/NeuralOps/triton_diag.log

cd /home/tahamajlesi.ut/NeuralOps
echo "=== Node Information ==="
hostname
date
nvidia-smi

export LD_LIBRARY_PATH=/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH

echo "=== DIAGNOSTIC: 26 TRITON BENCHMARK (neuralops_cu113) ==="
/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python benchmarks/benchmark_all_26_comprehensive.py --device cuda:0 --warmup 5 --repeat 10

echo "=== Job Finished ==="
date
