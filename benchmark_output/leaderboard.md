# 🏆 Attention Mechanism Benchmark Leaderboard

> Comprehensive evaluation of custom linear, low-rank, state-space, kernelized, and sparse attention architectures.

| Rank | Model | Score | Avg GLUE | SST-2 | QNLI | RTE | MRPC | CoLA | STS-B | Params | Throughput (seq=512) | VRAM (seq=512) | Stability | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **full** | **82.0** | 0.4324 | 0.5000 | N/A | N/A | 0.3648 | N/A | N/A | 66.96M | N/A | N/A | `FAIL` | `OK` |
| 2 | **svd_linear** | **76.69** | 0.3750 | 0.5625 | N/A | N/A | 0.1875 | N/A | N/A | 59.93M | N/A | N/A | `FAIL` | `OK` |

### 📊 Scoring Rubric Breakdown
- **Quality (40%)**: Ratio of model Average GLUE score against Full Standard Softmax Attention.
- **Efficiency (20%)**: Hardware execution throughput (tokens/sec) and GPU VRAM footprint at sequence length 512.
- **Stability (20%)**: Convergence consistency across multi-seed runs without majority-class collapse.
- **Robustness (20%)**: Hyperparameter sensitivity coefficient of variation (CV) across learning rates and batch sizes.

