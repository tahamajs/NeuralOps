# 2. Related Work

## 2.1 Low-Rank Attention Compression

**Linformer** (Wang et al., 2020) projects K and V to a fixed rank but requires 
retraining. **Nyströmformer** (Xiong et al., 2021) uses Nyström approximation. 
**SOA Fused Attention** differs by factorizing the projections themselves 
rather than the attention matrix.

**ALBERT** (Lan et al., 2020) uses cross-layer parameter sharing, a form of 
low-rank structure. We use within-layer factorization.

## 2.2 Weight Factorization

**SVD-based compression** is well studied (Denton et al., 2014; Zhang et al., 
2015). We apply it to attention projections with a specific twist: the 
factorization is **zero-shot** (no calibration data needed) and combined 
with LoRA adapters for recovery.

**Tensor decomposition** (Novikov et al., 2015; Garipov et al., 2016) uses 
higher-order structure. Our approach uses standard matrix SVD.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) freezes base weights and trains low-rank adapters. 
We combine LoRA with SVD factorization: the base factors are SVD-initialized 
and frozen, while LoRA provides the recovery signal.

**AdaLoRA** (Zhang et al., 2023) adaptively allocates rank. **DoRA** 
(Liu et al., 2024) decomposes magnitude and direction. Our approach is a 
complementary technique: compress first, then apply LoRA.

## 2.4 Attention Alternatives

**Multi-Query Attention** (Shazeer, 2019) shares KV heads. **Grouped-Query 
Attention** (Ainslie et al., 2023) groups KV heads. **Multi-Head Latent 
Attention** (DeepSeek-V2, 2024) projects KV to a low-rank latent. Our 
approach factorizes the projections themselves, which composes with GQA.

## 2.5 Position of Our Work

| Method | Calibration | Compresses | Zero-shot | GQA | RoPE |
|---|---|---|---|---|---|
| SVD (Vanilla) | No | All layers | Yes | Varies | Varies |
| A3 | Yes | All layers | No | No | No |
| AASVD | Yes | All layers | No | No | No |
| LoRA | No | Nothing | Yes | Yes | Yes |
| **SOA Fused** | **No** | **Q, K, V** | **Yes** | **Yes** | **Yes** |

**Key distinction.** SOA Fused Attention is the only method that combines:
- **Zero-shot** initialization (no calibration data)
- **Real** parameter reduction (not just adapter)
- **Attention-specific** targeting
- **Full architecture support** (GQA, RoPE, causal)

---

## Comparison Table

| Method | Reduction | Calibration | Retrain? | SST-2 acc |
|---|---|---|---|---|
| Baseline | 1× | — | — | 90.71% |
| DistilBERT (distill) | 1.6× | — | Yes | 90.7% |
| Vanilla SVD | 3× | No | LoRA | {{VANILLA_SVD_D}} |
| A3 | 3× | Yes | LoRA | {{A3_D}} |
| **SOA Fused** | **3×** | **No** | **LoRA** | **{{D_ACC}}** |
```

---