# 🏆 Attention Mechanism Benchmark Leaderboard

> Comprehensive evaluation of custom linear, low-rank, state-space, kernelized, and sparse attention architectures.

| Rank | Model | Score | Avg GLUE | SST-2 | QNLI | RTE | MRPC | CoLA | STS-B | Params | Throughput (seq=512) | VRAM (seq=512) | Stability | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **full** | **0.0** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 66.96M | N/A | N/A | `N/A` | `FAIL_SANITY` |
| 2 | **svd_linear** | **0.0** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | `N/A` | `FAIL_BUILD` |

### 📊 Scoring Rubric Breakdown
- **Quality (40%)**: Ratio of model Average GLUE score against Full Standard Softmax Attention.
- **Efficiency (20%)**: Hardware execution throughput (tokens/sec) and GPU VRAM footprint at sequence length 512.
- **Stability (20%)**: Convergence consistency across multi-seed runs without majority-class collapse.
- **Robustness (20%)**: Hyperparameter sensitivity coefficient of variation (CV) across learning rates and batch sizes.

