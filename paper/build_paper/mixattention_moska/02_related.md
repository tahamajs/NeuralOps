<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Token Pruning and KV Compression

**H2O** (Zhang et al., 2023). Keeps "heavy-hitter" tokens based on 
accumulated attention scores. Requires attention scores at each step, which 
FASA does not.

**StreamingLLM** (Xiao et al., 2024). Pins the first few tokens as attention 
sinks and keeps a sliding window. FASA's CLS-pinning is inspired by this.

**SnapKV** (Li et al., 2024). Uses attention scores from a prefix window to 
select which tokens to keep. FASA uses variance instead.

**Scissorhands** (Liu et al., 2023). Uses an importance criterion based on 
past attention patterns.

**PyramidKV** (Cai et al., 2024). Allocates different budgets to different 
layers.

## 2.2 Attention Sinks

**Xiao et al. (2024)** and **Gu et al. (2024)** show that transformers 
allocate disproportionate attention to the first few tokens. This is the 
basis for CLS-pinning in FASA.

## 2.3 Sparse Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**FASA differs from these** in that sparsity is **data-dependent** 
(variance of K) rather than architecturally fixed.

## 2.4 Position of Our Work

| Axis | Prior work | FASA |
|---|---|---|
| **Importance criterion** | Attention scores (H2O, SnapKV) | **Variance of K** |
| **Sink pinning** | First few tokens | **CLS only** |
| **Causal support** | Yes (with careful masking) | **No (skips pruning)** |
| **Training required** | No | No |
| **Parameters added** | None | LoRA only |

**Key distinction.** FASA's importance criterion is **cheaper** than H2O 
(no attention scores needed) but is **weaker** as a signal. Our ablation 
in Section 8 quantifies this.

---

## Comparison Table

| Method | Importance Signal | Requires Attention? | Sink Pin? | Causal? |
|---|---|---|---|---|
| H2O | Cumulative attention | Yes | Yes | Yes |
| StreamingLLM | Position | No | Yes (4) | Yes |
| SnapKV | Prefix attention | Yes | No | Yes |
| Scissorhands | Past attention | Yes | No | Yes |
| **FASA** | **Var(K)** | **No** | **Yes (1)** | **No** |

---