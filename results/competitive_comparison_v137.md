# 🚀 NeuralOps v1.3.7 Cross-Framework Competitive Benchmark

**Hardware Platform**: NVIDIA GeForce RTX 4070 Laptop GPU
**Batch Size**: 2 | **Embed Dim**: 768 | **Heads**: 12

| Seq Len (T) | Module | Framework | Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Speedup | Cosine Sim |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 512 | `pytorch_mha_baseline` | PyTorch Reference | 0.32 | 3,179,414.6 | 35.1 | **1.0x** | 1.0 |
| 512 | `gated_delta_net` | NeuralOps (Triton Fused) | 1.56 | 655,887.0 | 50.9 | **0.21x** | 1.0 |
| 512 | `gla_attention` | NeuralOps (Triton Fused) | 2.23 | 459,411.9 | 51.6 | **0.14x** | 1.0 |
| 512 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 3.09 | 331,806.0 | 120.4 | **0.1x** | 1.0 |
| 512 | `cosformer` | NeuralOps (Triton Fused) | 1.55 | 659,635.9 | 80.1 | **0.21x** | 1.0 |
| 512 | `minimal_svd` | NeuralOps (Triton Fused) | 0.59 | 1,744,396.4 | 25.1 | **0.55x** | 1.0 |
| 1024 | `pytorch_mha_baseline` | PyTorch Reference | 1.71 | 1,199,324.3 | 83.3 | **1.0x** | 1.0 |
| 1024 | `gated_delta_net` | NeuralOps (Triton Fused) | 3.08 | 665,347.7 | 82.8 | **0.55x** | 1.0 |
| 1024 | `gla_attention` | NeuralOps (Triton Fused) | 3.77 | 542,716.5 | 81.2 | **0.45x** | 1.0 |
| 1024 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 8.37 | 244,801.1 | 336.5 | **0.2x** | 1.0 |
| 1024 | `cosformer` | NeuralOps (Triton Fused) | 3.19 | 642,772.9 | 148.4 | **0.54x** | 1.0 |
| 1024 | `minimal_svd` | NeuralOps (Triton Fused) | 0.80 | 2,548,705.4 | 35.8 | **2.12x** | 1.0 |
| 2048 | `pytorch_mha_baseline` | PyTorch Reference | 5.24 | 782,026.7 | 255.6 | **1.0x** | 1.0 |
| 2048 | `gated_delta_net` | NeuralOps (Triton Fused) | 6.16 | 665,052.9 | 144.3 | **0.85x** | 1.0 |
| 2048 | `gla_attention` | NeuralOps (Triton Fused) | 6.97 | 587,375.7 | 137.3 | **0.75x** | 1.0 |
| 2048 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 25.59 | 160,061.6 | 1141.4 | **0.2x** | 1.0 |
| 2048 | `cosformer` | NeuralOps (Triton Fused) | 4.11 | 997,442.3 | 236.4 | **1.28x** | 1.0 |
| 2048 | `minimal_svd` | NeuralOps (Triton Fused) | 1.50 | 2,727,983.7 | 56.6 | **3.49x** | 1.0 |
| 4096 | `pytorch_mha_baseline` | PyTorch Reference | 19.12 | 428,518.6 | 912.6 | **1.0x** | 1.0 |
| 4096 | `gated_delta_net` | NeuralOps (Triton Fused) | 13.45 | 609,259.3 | 268.5 | **1.42x** | 1.0 |
| 4096 | `gla_attention` | NeuralOps (Triton Fused) | 13.44 | 609,455.5 | 252.0 | **1.42x** | 1.0 |
| 4096 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 87.42 | 93,713.8 | 4242.2 | **0.22x** | 1.0 |
| 4096 | `cosformer` | NeuralOps (Triton Fused) | 9.96 | 822,899.1 | 452.0 | **1.92x** | 1.0 |
| 4096 | `minimal_svd` | NeuralOps (Triton Fused) | 4.62 | 1,771,257.1 | 98.6 | **4.13x** | 1.0 |


*Generated automatically by `scripts/benchmarks/run_competitive_comparison.py`*
