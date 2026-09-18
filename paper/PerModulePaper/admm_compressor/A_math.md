# Appendix A. Augmented Lagrangian
$$\mathcal{L}_\rho(W, Z, Q, U_1, U_2) = f(W) + g(Z) + h(Q) + \frac{\rho_1}{2}\|W - Z + U_1\|_F^2 + \frac{\rho_2}{2}\|W - Q + U_2\|_F^2$$


## 1. Executive Abstract & Theoretical Motivation

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_compressor.md -->
Combining model pruning and weight quantization often leads to compounded precision degradation when executed sequentially. **ADMM Compressor** solves this by unifying structured magnitude pruning and uniform low-bit quantization under a single optimization formulation.

By formulating the pruning and quantization constraints simultaneously, ADMM Compressor identifies a sparse grid of discrete quantization levels. The algorithm first computes an absolute-magnitude threshold to enforce a target sparsity ratio $S$, then maps the retained non-zero weights into symmetric $b$-bit integer bins.

```
       Original Weight W
              │
              ▼
    ┌────────────────────┐
    │ Absolute Magnitude │ ──► Determine Threshold τ (Top-K Retained)
    │ Mask Generation    │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ Pruned Weights W_p │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ Symmetric b-Bit    │ ──► Scale Factor: 2^(b-1) - 1
    │ Quantization Grid  │
    └─────────┬──────────┘
              │
              ▼
   Compressed Quantized Weight W_q
```

---
