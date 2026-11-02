# Complete NeuralOps Research & Architecture Guide

A comprehensive architectural manual detailing:
1. **Mathematical Foundations**: SVD spectral decomposition, Fisher Information weighting, Eckart-Young-Mirsky theorem.
2. **3-Level Dispatch Hierarchy**: Native CUDA C++ (128-bit vector loads), autotuned Triton JIT, and vectorized PyTorch reference.
3. **Accuracy Healing**: 6-pillar knowledge distillation with Warmup-Stable-Decay (WSD) scheduling.
4. **Hardware Roofline Modeling**: Profiling on NVIDIA Ada Lovelace (SM 8.9), Hopper (SM 9.0), and Huawei Ascend NPUs.
