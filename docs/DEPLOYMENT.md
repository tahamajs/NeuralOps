# NeuralOps Production Deployment Guide

Deploying models optimized with NeuralOps across standard serving runtimes.

## Supported Export Formats
- **PyTorch / SafeTensors**: Native weights with custom dispatcher.
- **ONNX Runtime**: Graph export with fused kernels.
- **GGUF v3**: High-performance CPU/Apple Silicon execution with llama.cpp.
- **vLLM / SGLang**: RadixAttention and PagedAttention integration for low-rank and linear attention models.
