#!/usr/bin/env python3
import os
import sys

cuda_dir = 'neuralops/ops/cuda'
all_dirs = sorted([d for d in os.listdir(cuda_dir) if os.path.isdir(os.path.join(cuda_dir, d)) and not d.startswith('.') and not d.startswith('__')])

print("======================================================================")
print(f"  EXHAUSTIVE 10-PHASE AUDIT ACROSS ALL {len(all_dirs)} CUDA MODULES")
print("======================================================================")

total_cu = 0
total_cpp = 0
total_py = 0

for i in range(10):
    start = i * 13 + min(i, 4)
    end = (i + 1) * 13 + min(i + 1, 4)
    p_dirs = all_dirs[start:end]
    p_cu = 0
    p_cpp = 0
    p_py = 0
    for d in p_dirs:
        dp = os.path.join(cuda_dir, d)
        files = os.listdir(dp)
        p_cu += len([f for f in files if f.endswith('.cu')])
        p_cpp += len([f for f in files if f.endswith('.cpp')])
        p_py += len([f for f in files if f.endswith('.py') and not f.startswith('__')])
    total_cu += p_cu
    total_cpp += p_cpp
    total_py += p_py
    print(f"Phase {i+1:02d}: {len(p_dirs):2d} modules | {p_cu:2d} .cu | {p_cpp:2d} .cpp | {p_py:2d} .py | Range: {p_dirs[0]} -> {p_dirs[-1]}")

print("----------------------------------------------------------------------")
print(f"TOTAL:    {len(all_dirs)} modules | {total_cu} .cu | {total_cpp} .cpp | {total_py} .py")
print("======================================================================")
