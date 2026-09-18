# 5. Architecture and Implementation
Dense layers are replaced with `SVDLinear` modules ($y = (x B^	op) A^	op + b$). Supports Conv1D transposition for GPT-2 and preservation of LoRA adapters.


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/adasvd_compressor.md -->
### 3.1 Calibration Dataflow & Rank Assignment Pipeline

```
              Full Uncompressed Model
                         │
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Layer-wise SVD & Nuclear Norm Extraction     │
  │   For l = 1..L:                              │
  │     S = svdvals(W_l)                         │
  │     I_l = sum(S)                             │
  └──────────────────────┬───────────────────────┘
                         │  [Layer Importance Array I]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Global Importance Normalization              │
  │   alpha_l = I_l / sum(I_1..I_L)              │
  └──────────────────────┬───────────────────────┘
                         │  [Normalized Ratios alpha]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Adaptive Rank Allocation Calculation         │
  │   r_l = clamp(max_rank * (1 - ratio) * alpha)│
  └──────────────────────┬───────────────────────┘
                         │  [Allocated Ranks r_1..r_L]
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Loop over layers l = 1..L:                   │
  │  1. Extract initial U_r, S_r, Vt_r           │
  │  2. Execute T iterations of Alternating Refinement:│
  │      U_r <- W @ V_r @ (diag(S_r) + eps I)^-1  │
  │      Vt_r <- (U_r^T @ U_r + eps I)^-1 @ U_r^T @ W│
  │      S_r <- diag(U_r^T @ W @ V_r)            │
  │  3. Synthesize W_approx = U_r @ diag(S_r) @ Vt_r│
  └──────────────────────┬───────────────────────┘
                         │
                         ▼
               Compressed PyTorch Model
```

### 3.2 Detailed Step-by-Step Execution Sequence
1. **Spectral Profiling Phase**: Iterate over all `nn.Linear` layers in the model, execute standard CPU/GPU SVD to collect singular value vectors, and compute $\|W^{(l)}\|_*$.
2. **Global Budgeting Phase**: Aggregate total nuclear norm $\sum_{l=1}^L \mathcal{I}_l$. Compute relative ratios $\alpha_l$ and assign layer-specific ranks $r_l$.
3. **Iterative Refinement Phase**: For each linear layer:
Slice top-$r_l$ components from initial SVD matrices. Furthermore, Run $T = 5$ iterations of alternating matrix inverse updates to solve the regularized least-squares factor refinement problem. Concurrently, Multiply factors back to form the updated weight tensor $\hat{W}^{(l)}$ and overwrite `module.weight.data`.


---
