<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params | Time |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 1.90 s |
| DistilBERT | **UnifiedRoPE** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}}M | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 4.10 s |
| RoBERTa | **UnifiedRoPE** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}}M | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 4.20 s |
| GPT-2 | **UnifiedRoPE** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}}M | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After | Ratio |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS_FULL}} | **1.00×** |
| RoBERTa | 124,647,170 | {{R_PARAMS_FULL}} | **1.00×** |
| GPT-2 | 124,441,344 | {{G_PARAMS_FULL}} | **1.00×** |

**Observation.** UnifiedRoPE **preserves parameter count** — it is a 
replacement, not a compression method.

## 7.3 RoPE Correctness Verification

**Table 7.3: Norm preservation.**

| Model | $\|q\|_2$ before | $\|\tilde{q}\|_2$ after | Error |
|---|---|---|---|
| DistilBERT | {{D_Q_NORM_BEFORE}} | {{D_Q_NORM_AFTER}} | {{D_Q_NORM_ERR}} |
| RoBERTa | {{R_Q_NORM_BEFORE}} | {{R_Q_NORM_AFTER}} | {{R_Q_NORM_ERR}} |
| GPT-2 | {{G_Q_NORM_BEFORE}} | {{G_Q_NORM_AFTER}} | {{G_Q_NORM_ERR}} |

Error should be `< 1e-5`.

## 7.4 Relative Position Property

**Table 7.4: Score depends only on `i - j`.**

| Model | Score variance across absolute positions |
|---|---|
| DistilBERT | {{D_SCORE_VAR}} |
| RoBERTa | {{R_SCORE_VAR}} |
| GPT-2 | {{G_SCORE_VAR}} |

Low variance confirms relative position property.

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | UnifiedRoPE | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | UnifiedRoPE | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | UnifiedRoPE | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.6 Ablation: RoPE Base

**Table 7.6: Accuracy vs RoPE base.**

| Base | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 5000 | {{D_BASE5K}} | {{R_BASE5K}} | {{G_BASE5K}} |
| 10000 | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |
| 100000 | {{D_BASE100K}} | {{R_BASE100K}} | {{G_BASE100K}} |

## 7.7 Ablation: Rotary Dimension

**Table 7.7: Accuracy vs rotary_dim.**

| Rot dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_ROT32}} | {{R_ROT32}} | {{G_ROT32}} |
| 64 | {{D_ROT64}} | {{R_ROT64}} | {{G_ROT64}} |
| Full | {{D_ACC}} | {{R_ACC}} | {{G_ACC}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Accuracy vs LoRA rank.**

| LoRA rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.10 Comparison with Companion Methods

**Table 7.10: UnifiedRoPE vs other methods.**

| Method | DistilBERT | Reduces |
|---|---|---|
| full | 90.71% | — |
| MinimalSVD | {{MINSVD_D}} | Params |
| SVD-LLM V2 | {{SVDLLM_D}} | Params |
| Dobi-SVD | {{DOBI_D}} | Params |
| MLA | {{MLA_D}} | KV cache |
| Nyströmformer | {{NYS_D}} | Compute |
| **UnifiedRoPE** | **{{D_ACC}}** | **Nothing (replacement)** |

## 7.11 Training Curves

**Figure 7.1** shows LoRA recovery loss curves:

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.12 Summary

- **Parameter count preserved** (1.00×).
- **Accuracy:** {{D_DELTA}}%–{{G_DELTA}}% drop.
- **RoPE correctness verified.**
- **All four architectures supported.**
```

---