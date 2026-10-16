# 🚀 NeuralOps v1.3.7 Cross-Framework Competitive Benchmark

**Hardware Platform**: NVIDIA GeForce RTX 4070 Laptop GPU
**Batch Size**: 2 | **Embed Dim**: 768 | **Heads**: 12

| Seq Len (T) | Module | Framework | Latency (ms) | Throughput (tok/s) | Peak VRAM (MB) | Speedup | Cosine Sim |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 512 | `pytorch_mha_baseline` | PyTorch Reference | 0.32 | 3,161,437.7 | 35.1 | **1.0x** | 1.0 |
| 512 | `gated_delta_net` | NeuralOps (Triton Fused) | 1.00 | 1,027,196.6 | 44.4 | **0.32x** | 1.0 |
| 512 | `gla_attention` | NeuralOps (Triton Fused) | 0.49 | 2,067,147.6 | 41.2 | **0.65x** | 1.0 |
| 512 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 1.67 | 612,031.8 | 62.9 | **0.19x** | 1.0 |
| 512 | `cosformer` | NeuralOps (Triton Fused) | 1.60 | 639,572.1 | 88.0 | **0.2x** | 1.0 |
| 512 | `minimal_svd` | NeuralOps (Triton Fused) | 0.58 | 1,777,336.1 | 25.1 | **0.56x** | 1.0 |
| 1024 | `pytorch_mha_baseline` | PyTorch Reference | 1.72 | 1,194,479.8 | 83.3 | **1.0x** | 1.0 |
| 1024 | `gated_delta_net` | NeuralOps (Triton Fused) | 1.94 | 1,058,484.5 | 70.3 | **0.89x** | 1.0 |
| 1024 | `gla_attention` | NeuralOps (Triton Fused) | 0.95 | 2,150,457.8 | 65.5 | **1.8x** | 1.0 |
| 1024 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 2.76 | 741,303.5 | 94.2 | **0.62x** | 1.0 |
| 1024 | `cosformer` | NeuralOps (Triton Fused) | 3.40 | 602,152.2 | 156.5 | **0.5x** | 1.0 |
| 1024 | `minimal_svd` | NeuralOps (Triton Fused) | 0.55 | 3,741,446.1 | 35.8 | **3.14x** | 1.0 |
| 2048 | `pytorch_mha_baseline` | PyTorch Reference | 5.17 | 791,657.6 | 255.6 | **1.0x** | 1.0 |
| 2048 | `gated_delta_net` | NeuralOps (Triton Fused) | 3.68 | 1,112,094.3 | 120.2 | **1.4x** | 1.0 |
| 2048 | `gla_attention` | NeuralOps (Triton Fused) | 1.49 | 2,741,562.5 | 110.0 | **3.46x** | 1.0 |
| 2048 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 6.27 | 653,213.9 | 156.2 | **0.83x** | 1.0 |
| 2048 | `cosformer` | NeuralOps (Triton Fused) | 5.03 | 814,292.9 | 290.7 | **1.03x** | 1.0 |
| 2048 | `minimal_svd` | NeuralOps (Triton Fused) | 1.21 | 3,396,112.0 | 56.6 | **4.29x** | 1.0 |
| 4096 | `pytorch_mha_baseline` | PyTorch Reference | 19.12 | 428,551.8 | 912.6 | **1.0x** | 1.0 |
| 4096 | `gated_delta_net` | NeuralOps (Triton Fused) | 8.13 | 1,008,133.1 | 219.3 | **2.35x** | 1.0 |
| 4096 | `gla_attention` | NeuralOps (Triton Fused) | 3.23 | 2,538,085.5 | 198.5 | **5.92x** | 1.0 |
| 4096 | `hgrn2_gated_deltanet` | NeuralOps (Triton Fused) | 13.61 | 602,111.9 | 283.1 | **1.41x** | 1.0 |
| 4096 | `cosformer` | NeuralOps (Triton Fused) | 12.12 | 676,154.9 | 560.4 | **1.58x** | 1.0 |
| 4096 | `minimal_svd` | NeuralOps (Triton Fused) | 3.45 | 2,373,698.3 | 98.6 | **5.54x** | 1.0 |


*Generated automatically by `scripts/benchmarks/run_competitive_comparison.py`*
