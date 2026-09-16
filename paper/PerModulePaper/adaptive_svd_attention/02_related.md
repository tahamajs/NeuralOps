# 2. Related Work

## 2.1 Low-Rank Matrix Factorization

**Eckart–Young theorem** (Eckart & Young, 1936) establishes that truncated 
SVD is optimal in Frobenius norm at fixed rank. Our method uses this as a 
building block but **selects rank adaptively** via energy thresholding.

**Randomized SVD** (Halko et al., 2011) accelerates the decomposition. We 
use exact SVD because our matrices are modest size ($768 \times 768$).

**Energy-based rank selection** is used in PCA (Jolliffe, 2002) and latent 
semantic analysis (Deerwester et al., 1990). We apply it to attention 
projections for the first time.

## 2.2 Attention Compression

**SOA Fused Attention** (companion work) uses a **fixed** rank $k$ for 
Q/K/V. We differ in two ways: (a) dynamic rank per projection, (b) all four 
projections.

**A3** and **AASVD** (companion works) use activation-weighted SVD requiring 
calibration data. We are **zero-shot**.

**Linformer** (Wang et al., 2020) projects K, V to a fixed rank at the 
sequence level. Our projection is at the feature level.

## 2.3 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2021) adds trainable low-rank deltas to frozen weights. 
Our method adds LoRA on top of factorized SVD — a two-level low-rank 
structure.

**AdaLoRA** (Zhang et al., 2023) also allocates rank adaptively, but for the 
adapter, not for the base factorization.

## 2.4 T5 Relative Attention

**T5** (Raffel et al., 2020) uses a learned relative position bias added to 
attention scores. This mechanism is orthogonal to our factorization but 
requires special handling in the weight transfer.

## 2.5 Cross-Attention

Encoder-decoder models (T5, BART) have cross-attention with separate K, V 
projections from the encoder. Our method supports cross-attention because 
all projections are factorized.

## 2.6 Position of Our Work

| Method | Rank | Calibration | Projections | T5 | Cross-attn |
|---|---|---|---|---|---|
| SOA Fused | Fixed | No | Q/K/V | No | No |
| A3 | Fixed | Yes | Q/K/V/O | No | No |
| AASVD | Fixed | Yes | Q/K/V/O | No | No |
| **Adaptive SVD** | **Dynamic** | **No** | **Q/K/V/O** | **Yes** | **Yes** |

**Key distinction.** Adaptive SVD Attention is the only method that:
- Uses **dynamic rank per projection**
- Is **zero-shot** (no calibration)
- Supports **T5 relative attention**
- Handles **encoder-decoder cross-attention**

---

## Comparison Table

| Method | Rank | Energy τ | Reduction | Retrain? | SST-2 acc |
|---|---|---|---|---|---|
| Baseline | — | — | 1× | — | 90.71% |
| SOA Fused (k=128) | Fixed | — | 2× | LoRA | {{SOA_D}} |
| A3 | Fixed | — | 3× | LoRA | {{A3_D}} |
| **Adaptive SVD** | **Dynamic** | **0.92** | **{{D_RATIO}}×** | **LoRA** | **{{D_ACC}}** |
```

---