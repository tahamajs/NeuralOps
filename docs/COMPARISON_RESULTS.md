# Benchmark Comparison Results

### NeuralOps vs Existing Libraries

Benchmark configuration:
- Batch Size ($B$): 2
- Sequence Length ($N$): 512
- Head Dimension ($D$): 64
- Number of Heads ($H$): 4
- Execution Device: CPU / Metal (pure PyTorch fallback without CUDA FLA hardware)

| Module | NeuralOps (ms) | FLA (ms) | Ratio |
|:---|:---:|:---:|:---:|
| `gated_linear_attention` | 13.17 ms | N/A (requires CUDA) | - |
| `decay_linear_attention` | 8.20 ms | N/A (requires CUDA) | - |

> **Note**: FLA requires native NVIDIA CUDA drivers and Triton Linux kernels to compile its custom C++/CUDA extensions. NeuralOps runs seamlessly across CPU, Apple Silicon (MPS), and CUDA out of the box with automatic fallbacks.
