# NeuralOps Native CUDA C++ Subsystem (`neuralops.ops.cuda`)

This package provides native, production-grade CUDA C++ implementations of hot-path operations for NeuralOps across 112 modular kernel packages (114 native implementations).

## Architecture

Every operation adheres to the strict 3-level fallback contract:
1. **CUDA C++**: Native hand-tuned kernels with asynchronous memory pipelines, Tensor Core MMA, explicit shared memory double-buffering, and warp shuffle primitives.
2. **Triton JIT**: High-performance Triton kernels (26 canonical modules housing 62 unified operators across 4 tiers).
3. **PyTorch Reference**: Numerical ground truth with guaranteed execution across all platforms (CPU, Apple Silicon MPS).

## Build Requirements
- NVIDIA GPU with Compute Capability (SM) >= 7.0 (Volta, Turing, Ampere, Ada Lovelace, Hopper, Blackwell).
- CUDA Toolkit with `nvcc` installed and in `$PATH` (or specified via `CUDA_HOME`).
