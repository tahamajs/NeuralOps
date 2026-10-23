# NeuralOps CUDA C++ Hot-Path Subsystem (`neuralops.core.cuda_kernels`)

> **Note (v1.6.0 Tier Migration)**: Canonical native CUDA C++ kernel packages are housed in [`neuralops.ops.cuda`](../../ops/cuda/). This module provides backward-compatibility dispatch shims.

This package provides native CUDA C++ implementations of hot-path operations for NeuralOps across 112 modular packages (114 kernel implementations).

## Architecture

Every operation follows a strict 3-level fallback architecture:
1. **CUDA C++**: Native hand-tuned kernels with asynchronous memory pipelines, Tensor Core MMA, explicit shared memory double-buffering, and dynamic control flow.
2. **Triton JIT**: High-performance Triton kernels (26 canonical modules, 62 unified operators).
3. **PyTorch Reference**: Numerical ground truth with guaranteed execution across all platforms (CPU, MPS).

## Build Requirements
- NVIDIA GPU with Compute Capability (SM) >= 7.0 (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell).
- CUDA Toolkit with `nvcc` installed and in `$PATH` (or specified via `CUDA_HOME`).
