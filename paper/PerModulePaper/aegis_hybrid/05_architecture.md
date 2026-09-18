# 5. Architecture and Implementation


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```

Projections use `FactorizedLinear` ($r=128, r_{	ext{lora}}=32$). The gating network dynamically arbitrates between local and global representations.


## 4. Step-by-Step Tensor Execution Trace & Dimensions

<!-- source: Codes/frontend/dist/Docs/content/attentions/aegis_hybrid.md -->
```
Given: Batch Size B=8, Sequence Length L=128, Hidden Dim D=768, Heads H=12, Head Dim d_h=64, Window W=128

1. Input Tensor:
   X ∈ ℝ^{8 × 128 × 768}

2. Projection Stage (FactorizedLinear with LoRA):
   q = X W_Q1 W_Q2 + (α/r) X A_Q B_Q + b_Q    ──>  [8, 128, 768]
   k = X W_K1 W_K2 + (α/r) X A_K B_K + b_K    ──>  [8, 128, 768]
   v = X W_V1 W_V2 + (α/r) X A_V B_V + b_V    ──>  [8, 128, 768]

3. Head Splitting & Reshaping:
   Q: [8, 12, 128, 64]
   K: [8, 12, 128, 64]
   V: [8, 12, 128, 64]

4. Positional Embedding Application:
   Q_rope, K_rope = apply_rotary_emb(Q, K, position_ids)  ──>  [8, 12, 128, 64]

5. Local Window Attention Pathway:
   scores = (Q_rope @ K_rope^T) / √64                      ──>  [8, 12, 128, 128]
   scores = scores.masked_fill(~band_mask, -10000.0)      ──>  [8, 12, 128, 128]
   attn_local = softmax(scores, dim=-1)                   ──>  [8, 12, 128, 128]
   out_local = attn_local @ V                             ──>  [8, 128, 12, 64]

6. Global Linear Pathway (Bidirectional Encoders):
   phi_Q = ELU(Q_rope) + 1.0                              ──>  [8, 12, 128, 64]
   phi_K = ELU(K_rope) + 1.0                              ──>  [8, 12, 128, 64]
   KV = phi_K^T @ V                                       ──>  [8, 12, 64, 64]
   K_sum = phi_K.sum(dim=2)                               ──>  [8, 12, 64]
   out_global = (phi_Q @ KV) / (phi_Q @ K_sum + ε)        ──>  [8, 128, 12, 64]

7. Gated Convex Fusion:
   α = σ(X @ W_fusion + 1.8)                              ──>  [8, 128, 12, 1]
   fused = α * out_local + (1 - α) * out_global           ──>  [8, 128, 12, 64]

8. Concatenation & Output Factorized Projection:
   fused = fused.reshape(8, 128, 768)
   Output = fused W_O1 W_O2 + (α/r) fused A_O B_O + b_O   ──>  [8, 128, 768]
```

---
