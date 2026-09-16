# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) introduced the modern 
formulation with $\phi(x) = \text{elu}(x) + 1$. It achieves 
$\mathcal{O}(L)$ time but uses no decay.

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features. **cosFormer** (Luo et al., 2022) uses a cos-based feature map.

**GLA** (Yang et al., 2024) adds **learned decay** to linear attention, 
similar in spirit to our work. However, GLA uses **fixed** per-head decay 
rates, not input-dependent ones.

**Mamba** (Gu & Dao, 2023) uses **selective** SSM where the transition 
matrix is input-dependent. Our decay is analogous: $\lambda_{h,t}$ depends 
on the input.

## 2.2 Exponential Decay in RNNs

**HGRN** (Qin et al., 2023) uses a bounded forget gate in an element-wise 
recurrence. **RetNet** (Sun et al., 2023) uses a fixed exponential decay 
per head. **RWKV** (Peng et al., 2023) uses data-dependent decay in a 
linear-attention-like formulation.

Our decay is per-head (like RetNet) and data-dependent (like HGRN), which 
is a strict generalization.

## 2.3 Chunked Linear Attention

**FlashLinearAttention** (Yang & Zhang, 2024) uses chunked algorithms for 
linear attention. **Mamba** uses a parallel scan over chunks. **DeltaNet** 
uses chunked parallelization for the delta rule.

Our chunked scan is a simpler variant specialized for exponential decay:
the state $S$ is decayed by a scalar between chunks, and the intra-chunk 
computation is a small dense causal attention.

## 2.4 Spherical and Other Feature Maps

**CHELA-SLAY** (companion work) uses spherical normalization + Yat-Kernel. 
**Zhang et al. (2024)** propose $\phi(x) = \text{elu}(x) + 1$ as a 
general-purpose feature map. We adopt ELU+1 for its simplicity and 
positivity.

## 2.5 Position of Our Work

| Method | Complexity | Decay | Data-dep | O(1) causal mem | Cross-attn |
|---|---|---|---|---|---|
| Linear Transformer | $\mathcal{O}(L)$ | None | ❌ | ❌ | ❌ |
| Performer | $\mathcal{O}(L)$ | None | ❌ | ❌ | ❌ |
| GLA | $\mathcal{O}(L)$ | Fixed | ❌ | Partial | ❌ |
| RetNet | $\mathcal{O}(L)$ | Fixed | ❌ | ✅ | ❌ |
| Mamba | $\mathcal{O}(L)$ | Fixed + selective | Partial | ✅ | ❌ |
| CHELA-SLAY | $\mathcal{O}(L^2)$ | None | ❌ | ❌ | ❌ |
| **DecayLinear** | **$\mathcal{O}(L)$** | **Data-dep** | ✅ | ✅ | ✅ |

**Key distinction.** DecayLinearAttention combines:
- **Data-dependent** exponential decay
- **O(1) causal memory** via chunked scan
- **Cross-attention** support

No prior method combines all three.

---

## Comparison Table

| Method | Decay | Memory (L=1024) | SST-2 | Cross-attn |
|---|---|---|---|---|
| Baseline MHSA | None | ~3 MB | 90.71% | ✅ |
| Linear Transformer | None | ~0.3 MB | {{LT_D}} | ❌ |
| GLA | Fixed | ~0.3 MB | {{GLA_D}} | ❌ |
| CHELA-SLAY | None | ~65 MB | {{CS_D}} | ❌ |
| **DecayLinear** | **Data-dep** | **~1.5 MB** | **{{D_ACC}}** | **✅** |

---