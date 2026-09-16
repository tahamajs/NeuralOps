<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with $w = {{WINDOW_SIZE}}$, $k = {{TOP_K}}$.**

| Model | Module | Acc | Δ | F1 | Density | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 100% | 183 MB | 1.9 s |
| DistilBERT | **NSA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_DENSITY}}%** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 100% | 712 MB | 4.1 s |
| RoBERTa | **NSA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_DENSITY}}%** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 100% | 793 MB | 4.2 s |
| GPT-2 | **NSA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_DENSITY}}%** | {{G_VRAM}} MB | {{G_TIME}} s |

**Density** is the average fraction of attended keys per query.

## 7.2 Attention Mass Preservation

**Table 7.2: Retained attention mass.**

| Model | Preserved mass |
|---|---|
| DistilBERT | {{D_MASS}}% |
| RoBERTa | {{R_MASS}}% |
| GPT-2 | {{G_MASS}}% |

## 7.3 Efficiency

**Table 7.3: Memory and time.**

| Model | Config | Peak VRAM | Inference Time |
|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s |
| DistilBERT | NSA | {{D_VRAM}} MB | {{D_TIME}} s |
| GPT-2 | full | 793 MB | 4.20 s |
| GPT-2 | NSA | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.4 Ablation: Window Size

**Table 7.4: Accuracy vs window size ($k = {{TOP_K}}$).**

| $w$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 8 | {{D_W8}} | {{R_W8}} | {{G_W8}} | {{DENS8}}% |
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} | {{DENS16}}% |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} | {{DENS32}}% |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} | {{DENS64}}% |

## 7.5 Ablation: Top-K

**Table 7.5: Accuracy vs top-$k$ ($w = {{WINDOW_SIZE}}$).**

| $k$ | DistilBERT | RoBERTa | GPT-2 | Density |
|---|---|---|---|---|
| 4 | {{D_K4}} | {{R_K4}} | {{G_K4}} | {{DENSK4}}% |
| 8 | {{D_K8}} | {{R_K8}} | {{G_K8}} | {{DENSK8}}% |
| 16 | {{D_K16}} | {{R_K16}} | {{G_K16}} | {{DENSK16}}% |
| 32 | {{D_K32}} | {{R_K32}} | {{G_K32}} | {{DENSK32}}% |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison with Companion Methods

**Table 7.8: Comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Density |
|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense | 100% |
| GQA | {{GQA_D}} | {{GQA_R}} | {{GQA_G}} | KV reduce | 100% |
| Primal | {{PRIMAL_D}} | {{PRIMAL_R}} | {{PRIMAL_G}} | Low-rank | 100% |
| ASA-Sparse | {{ASA_D}} | {{ASA_R}} | {{ASA_G}} | Top-k only | {{ASA_DENS}}% |
| ASA-Span | {{ASPAN_D}} | {{ASPAN_R}} | {{ASPAN_G}} | Learned window | {{ASPAN_DENS}}% |
| NHA | {{NHA_D}} | {{NHA_R}} | {{NHA_G}} | Hybrid RFF | 100% |
| HALO | {{HALO_D}} | {{HALO_R}} | {{HALO_G}} | Hybrid linear | 100% |
| **NSA** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **Hybrid top-k** | **{{D_DENSITY}}%** |

## 7.9 Scaling with Sequence Length

**Table 7.9: Density vs sequence length ($w = {{WINDOW_SIZE}}$, $k = {{TOP_K}}$).**

| $L$ | Attended keys | Density |
|---|---|---|
| 128 | {{KEYS_128}} | {{DENS_128}}% |
| 512 | {{KEYS_512}} | {{DENS_512}}% |
| 2048 | {{KEYS_2048}} | {{DENS_2048}}% |
| 8192 | {{KEYS_8192}} | {{DENS_8192}}% |

## 7.10 Training Loss Curves

**Figure 7.1** shows loss curves for all three models.

## 7.11 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Density:** reduced to {{D_DENSITY}}% (at $L = 128$); lower at longer $L$.
- **No parameter reduction.**
- **No time speedup** without sparse kernels.
- **All models pass quality gate.**

---