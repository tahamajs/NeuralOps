# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Path | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | dense | 183 MB |
| DistilBERT | **NEXUS** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | sparse | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | dense | 712 MB |
| RoBERTa | **NEXUS** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | sparse | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | dense | 793 MB |
| GPT-2 | **NEXUS** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **dense fallback** | {{G_VRAM}} MB |

**Key observations:**

1. DistilBERT and RoBERTa use the sparse path.
2. GPT-2 falls back to dense (causal).
3. NEXUS provides **no compute savings on GPT-2**.

## 7.2 Ablation: Components

**Table 7.2: Effect of each NEXUS component.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full NEXUS | {{D_FULL}} | {{R_FULL}} | {{G_FULL}} |
| − XAttention | {{D_NOXA}} | {{R_NOXA}} | — |
| − Seer gate | {{D_NOSEER}} | {{R_NOSEER}} | — |
| − Stripe | {{D_NOSTRIPE}} | {{R_NOSTRIPE}} | — |
| − Softpick | {{D_NOSOFTPICK}} | {{R_NOSOFTPICK}} | — |
| Dense only | {{D_DENSE}} | {{R_DENSE}} | {{G_DENSE}} |

## 7.3 Ablation: Block Size

**Table 7.3: Accuracy vs block size.**

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | {{D_B8}} | {{R_B8}} |
| 16 | {{D_B16}} | {{R_B16}} |
| 32 | {{D_B32}} | {{R_B32}} |
| 64 | {{D_B64}} | {{R_B64}} |

## 7.4 Ablation: Top-K Blocks

**Table 7.4: Accuracy vs $k$.**

| $k$ | DistilBERT | RoBERTa |
|---|---|---|
| 1 | {{D_K1}} | {{R_K1}} |
| 2 | {{D_K2}} | {{R_K2}} |
| 4 | {{D_K4}} | {{R_K4}} |
| 8 | {{D_K8}} | {{R_K8}} |

## 7.5 Ablation: Softpick Temperature

**Table 7.5: Accuracy vs softpick temperature.**

| $T$ | DistilBERT | RoBERTa |
|---|---|---|
| 0.5 | {{D_T05}} | {{R_T05}} |
| 1.0 | {{D_T10}} | {{R_T10}} |
| 2.0 | {{D_T20}} | {{R_T20}} |

## 7.6 Ablation: Sparse Min Seq Len

**Table 7.6: Effect of `sparse_min_seq_len`.**

| min_seq_len | DistilBERT | RoBERTa |
|---|---|---|
| 4 | {{D_S4}} | {{R_S4}} |
| 8 | {{D_S8}} | {{R_S8}} |
| 16 | {{D_S16}} | {{R_S16}} |
| 32 | {{D_S32}} | {{R_S32}} |

## 7.7 Efficiency Profile

**Table 7.7: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | NEXUS | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | NEXUS (dense) | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.8 Router Statistics

**Table 7.8: Seer gate behavior.**

| Model | Mean gate | Mean kept blocks |
|---|---|---|
| DistilBERT | {{D_MEANGATE}} | {{D_MEANKEEP}} |
| RoBERTa | {{R_MEANGATE}} | {{R_MEANKEEP}} |

## 7.9 Component Activation Summary

**Table 7.9: Which components are active per model.**

| Component | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| XAttention | ✓ | ✓ | ✗ (dense fallback) |
| Seer gate | ✓ | ✓ | ✗ |
| DuSA | ✓ | ✓ | ✗ |
| Stripe | ✗ ($N \leq 256$) | ✗ | ✗ |
| RFF rectification | ✗ (default) | ✗ | ✗ |
| Softpick | ✓ | ✓ | ✗ |

**Critical observation.** Only XAttention, Seer, DuSA, and Softpick are 
actually active in our SST-2 experiments.

## 7.10 Summary

- NEXUS provides sub-quadratic attention for encoders.
- No benefit on causal decoders.
- Several components (stripe, RFF) are inactive under our setup.

---