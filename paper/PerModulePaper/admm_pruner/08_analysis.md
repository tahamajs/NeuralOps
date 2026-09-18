# 8. Detailed Analysis
Pruning 72 layers on RoBERTa and 36 on DistilBERT at 50% sparsity maintains strong task performance (84.29% and 80.62%). On GPT-2, autoregressive generation requires smaller pruning ratios to avoid degradation.


## 4. Parameter Savings & Memory Footprint Profiling

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_pruner.md -->
### 4.1 Memory Footprint Profiling

| Metric | Formula | Example (LLaMA-7B Gate Proj: $N=11008, M=4096$) |
| :--- | :--- | :--- |
| **Dense Weights** | $N \times M \times 2 \text{ bytes}$ | $90.18 \text{ MB}$ |
| **Sparse Weights (CSR/COO)** | $(1 - S) \cdot N \cdot M \cdot (2 + 4) \text{ bytes}$ | $40.58 \text{ MB}$ ($S=0.50$, COO 32-bit idx) |
| **Calibration Memory Peak** | $W + Z + U = 3 \times N \times M \times 4 \text{ bytes}$ | $541.09 \text{ MB}$ (FP32 precision) |
| **Inference VRAM Savings** | $1 - (1 - S) \cdot (1 + \text{Overhead})$ | $\approx 45.2\%$ Memory Reduction |

---


## 5. Precision Loss, Perplexity & Sensitivity Bounds

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_pruner.md -->
### 5.1 Perplexity Sensitivity Analysis

The theoretical error bound on layer output $\Delta Y = (W - Z) X$ is governed by the spectral norm of the constraint residual:

$$\| \Delta Y \|_2 \le \| W - Z \|_2 \cdot \| X \|_2 \le \sqrt{S \cdot N \cdot M} \cdot \max_{i,j} |U_{i,j}| \cdot \| X \|_2$$

Because $U$ accumulates residual drift, larger values of penalty $\rho$ damp $U$, maintaining tighter output bounds.

### 5.2 Failure Cases & Edge Conditions

Specifically, regarding \textbf{Ill-Conditioned Layers}, in layers with heavy-tailed weight distributions (e.g., output projections), percentile masking may prune critical outlier channels. Furthermore, regarding \textbf{Small Iteration Limits}, setting $K < 3$ prevents dual variable $U$ from stabilizing, reducing ADMM to naive magnitude pruning.


---
