# Live GPU Accuracy-Recovery Proof Benchmark (RTX 4070)

| Protocol | Cosine Similarity to Teacher | KL Divergence to Teacher | Peak VRAM (MB) | Execution Time (s) |
|---|:---:|:---:|:---:|:---:|
| **Zero-Shot Post-Surgery (Before Recovery)** | 0.5461 | 2.9729 | - | - |
| **Naive Cross-Entropy Fine-Tuning (50 steps)** | 0.2355 | 4.4677 | 1021.4 | 1.89s |
| **6-Pillar MOHAWK + PiSSA + DKD Engine (50 steps)** | **0.9043** | **0.5988** | **2267.6** | **5.88s** |

