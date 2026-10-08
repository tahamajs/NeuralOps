# Memory Profiling Report

### NeuralOps Memory Allocation Benchmarks

Memory measured during forward pass evaluation using `tracemalloc` (CPU peak) and PyTorch memory tracking:
- Batch Size ($B$): 2
- Sequence Length ($N$): 512
- Hidden Dimension: 256
- Number of Heads: 4

| Attention Mechanism | CPU Peak Memory (MB) | CUDA Peak Memory (MB) |
|:---|:---:|:---:|
| `svd_linear_attention` | < 0.1 MB | N/A (GPU) |
| `cosformer` | < 0.1 MB | N/A (GPU) |
| `gqa_attention` | < 0.1 MB | N/A (GPU) |

All tested attention mechanisms demonstrate negligible CPU memory overhead under pure PyTorch execution.
