#!/bin/bash
#PBS -N surgery_qwen_gqa
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /home/tahamajlesi.ut/NeuralOps/surgery_qwen_gqa.log

cd /home/tahamajlesi.ut/NeuralOps
echo "=== Node Information ==="
hostname
date
nvidia-smi

export LD_LIBRARY_PATH=/share/apps/gcc/gcc-8.2.0/lib64:/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH

echo "=== PyTorch GPU Test ==="
/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== Running PyTest on Surgery Qwen GQA ==="
/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python -m pytest tests/gpu_exhaustive/test_surgery_qwen_gqa.py -v --tb=long

echo "=== Job Finished ==="
date
