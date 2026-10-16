# NeuralOps CUDA C++ Hot-Path Subsystem

This package provides native, production-grade CUDA C++ implementations of hot-path operations for NeuralOps (SVD_linear_Attention v1.3.7+).

## Architecture

Every operation follows a strict 3-level fallback architecture:
1. **CUDA C++**: Native hand-tuned kernels with asynchronous memory pipelines, Tensor Core MMA, explicit shared memory double-buffering, and dynamic control flow.
2. **Triton JIT**: High-performance Triton kernels (26 canonical modules).
3. **PyTorch Reference**: Numerical ground truth with guaranteed execution across all platforms.

## Build Requirements
- NVIDIA GPU with Compute Capability (SM) >= 7.0 (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell).
- CUDA Toolkit with `nvcc` installed and in `$PATH` (or specified via `CUDA_HOME`).
- Build artifacts are cached in `/mnt/c/linux_cache/torch_extensions` to prevent root filesystem pressure.
