# 🚀 NeuralOps v1.3.7 Cross-Framework Competitive Benchmark

**Hardware Platform**: NVIDIA GeForce RTX 4070 Laptop GPU
**Batch Size**: 2 | **Embed Dim**: 768 | **Heads**: 12

| Seq Len (T) | Module | Framework | Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Speedup | Cosine Sim |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 512 | `pytorch_mha_baseline` | PyTorch Reference | 0.32 | 3,192,931.6 | 35.1 | **1.0x** | 1.0 |
| 512 | `gated_delta_net` | NeuralOps (Triton Fused) | 1.56 | 655,558.8 | 50.9 | **0.21x** | 1.0 |
| 512 | `gla_attention` | NeuralOps (Triton Fused) | 2.17 | 472,202.7 | 51.6 | **0.15x** | 1.0 |
| 512 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 3.08 | 332,280.8 | 120.4 | **0.1x** | 1.0 |
| 512 | `cosformer` | NeuralOps (Triton Fused) | 1.54 | 665,179.6 | 80.1 | **0.21x** | 1.0 |
| 512 | `minimal_svd` | NeuralOps (Triton Fused) | 0.38 | 2,667,239.0 | 25.1 | **0.84x** | 1.0 |
| 1024 | `pytorch_mha_baseline` | PyTorch Reference | 1.70 | 1,205,684.8 | 83.3 | **1.0x** | 1.0 |
| 1024 | `gated_delta_net` | NeuralOps (Triton Fused) | 3.06 | 668,306.9 | 82.8 | **0.55x** | 1.0 |
| 1024 | `gla_attention` | NeuralOps (Triton Fused) | 3.74 | 547,281.6 | 81.2 | **0.45x** | 1.0 |
| 1024 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 7.94 | 257,842.2 | 336.5 | **0.21x** | 1.0 |
| 1024 | `cosformer` | NeuralOps (Triton Fused) | 3.20 | 639,723.6 | 148.4 | **0.53x** | 1.0 |
| 1024 | `minimal_svd` | NeuralOps (Triton Fused) | 0.61 | 3,366,497.8 | 35.8 | **2.79x** | 1.0 |
| 2048 | `pytorch_mha_baseline` | PyTorch Reference | 5.24 | 782,350.1 | 255.6 | **1.0x** | 1.0 |
| 2048 | `gated_delta_net` | NeuralOps (Triton Fused) | 6.15 | 665,736.5 | 144.3 | **0.85x** | 1.0 |
| 2048 | `gla_attention` | NeuralOps (Triton Fused) | 6.83 | 599,894.7 | 137.3 | **0.77x** | 1.0 |
| 2048 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 25.58 | 160,105.8 | 1141.4 | **0.2x** | 1.0 |
| 2048 | `cosformer` | NeuralOps (Triton Fused) | 4.10 | 999,314.4 | 236.4 | **1.28x** | 1.0 |
| 2048 | `minimal_svd` | NeuralOps (Triton Fused) | 1.47 | 2,786,132.4 | 56.6 | **3.56x** | 1.0 |
| 4096 | `pytorch_mha_baseline` | PyTorch Reference | 19.12 | 428,477.4 | 912.6 | **1.0x** | 1.0 |
| 4096 | `gated_delta_net` | NeuralOps (Triton Fused) | 13.43 | 610,101.1 | 268.5 | **1.42x** | 1.0 |
| 4096 | `gla_attention` | NeuralOps (Triton Fused) | 13.41 | 611,062.1 | 252.0 | **1.43x** | 1.0 |
| 4096 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 87.39 | 93,735.0 | 4242.2 | **0.22x** | 1.0 |
| 4096 | `cosformer` | NeuralOps (Triton Fused) | 9.95 | 823,451.0 | 452.0 | **1.92x** | 1.0 |
| 4096 | `minimal_svd` | NeuralOps (Triton Fused) | 4.62 | 1,774,424.8 | 98.6 | **4.14x** | 1.0 |


*Generated automatically by `scripts/benchmarks/run_competitive_comparison.py`*
