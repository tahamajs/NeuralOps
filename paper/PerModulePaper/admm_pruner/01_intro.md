# 1. Introduction
Structured pruning enforces hardware-efficient patterns (such as NVIDIA 2:4 sparsity) but can severely disrupt intermediate representations. `admm_pruner` introduces dynamic $ho$ adjustment and layer-wise feature distillation to steer weight updates towards sparse projection manifolds without sacrificing task accuracy.
