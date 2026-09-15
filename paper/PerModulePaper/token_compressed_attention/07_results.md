# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | KV Len | ρ | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 128 | — | 183 MB |
| DistilBERT | **TCA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_KVLEN}} | {{RATIO}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 128 | — | 712 MB |
| RoBERTa | **TCA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_KVLEN}} | {{RATIO}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 128 | — | 793 MB |
| GPT-2 | **TCA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_KVLEN}} | {{RATIO}} | {{G_VRAM}} MB |

**Key observations:**

1. KV sequence length reduced from 128 to **{{K_NEW}}** (~{{RATIO}}× reduction).
2. Attention compute reduced by **{{COMPUTE_REDUCTION}}%**.
3. Accuracy drop of **{{D_DELTA}}%** on DistilBERT.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts (per model).**

| Model | Baseline | TCA | Change | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% | LoRA + score_net |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% | LoRA + score_net |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% | LoRA + score_net |

**We state explicitly:** TCA does **not** reduce parameter count. The 
increase is from LoRA adapters and the score network.

## 7.3 KV Cache and Compute

**Table 7.3: Inference efficiency.**

| Metric | Baseline | TCA | Savings |
|---|---|---|---|
| KV length | 128 | {{K_NEW}} | {{KV_PCT}}% |
| KV cache bytes | {{KV_BASE}} | {{KV_TCA}} | {{KV_PCT}}% |
| Attention FLOPs | {{FLOP_BASE}} | {{FLOP_TCA}} | {{FLOP_PCT}}% |
| Inference time | {{T_BASE}} s | {{T_TCA}} s | {{TIME_PCT}}% |
| Peak VRAM | {{VRAM_BASE}} MB | {{VRAM_TCA}} MB | {{VRAM_PCT}}% |

## 7.4 Ablation: Compression Ratio ρ

**Table 7.4: Accuracy vs compression ratio.**

| ρ | DistilBERT | KV Len | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.9 | {{D_R09}} | 116 | {{R_R09}} | {{G_R09}} |
| 0.7 | {{D_R07}} | 90 | {{R_R07}} | {{G_R07}} |
| 0.5 | {{D_R05}} | 64 | {{R_R05}} | {{G_R05}} |
| 0.3 | {{D_R03}} | 39 | {{R_R03}} | {{G_R03}} |
| 0.1 | {{D_R01}} | 13 | {{R_R01}} | {{G_R01}} |

## 7.5 Ablation: Score Net

**Table 7.5: Effect of score net.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| TCA (norm + score_net) | {{D_NORM_LEARN}} | {{R_NORM_LEARN}} | {{G_NORM_LEARN}} |
| TCA (norm only) | {{D_NORM}} | {{R_NORM}} | {{G_NORM}} |
| TCA (score_net only) | {{D_LEARN}} | {{R_LEARN}} | {{G_LEARN}} |
| Random top-k | {{D_RANDOM}} | {{R_RANDOM}} | {{G_RANDOM}} |

**Observation.** {{SCORE_NET_OBS}}

## 7.6 Ablation: [CLS] Preservation

**Table 7.6: Effect of pinning token 0.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With [CLS] pinned | {{D_CLS}} | {{R_CLS}} | {{G_CLS}} |
| Without pinning | {{D_NOCLS}} | {{R_NOCLS}} | {{G_NOCLS}} |
| Δ | {{D_CLS_DELTA}} | {{R_CLS_DELTA}} | {{G_CLS_DELTA}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Scaling with Sequence Length

**Table 7.8: Inference time vs sequence length (GPT-2).**

| L | Baseline | TCA (ρ=0.5) | Speedup |
|---|---|---|---|
| 128 | {{B128}} s | {{T128}} s | {{S128}}× |
| 256 | {{B256}} s | {{T256}} s | {{S256}}× |
| 512 | {{B512}} s | {{T512}} s | {{S512}}× |
| 1024 | {{B1024}} s | {{T1024}} s | {{S1024}}× |

## 7.9 Comparison with Eviction Baselines

**Table 7.9: TCA vs H2O-style and StreamingLLM-style.**

| Method | DistilBERT | RoBERTa | GPT-2 | Learnable |
|---|---|---|---|---|
| Random top-k | {{RANDOM_D}} | {{RANDOM_R}} | {{RANDOM_G}} | No |
| H2O-style | {{H2O_D}} | {{H2O_R}} | {{H2O_G}} | No |
| StreamingLLM-style | {{SLM_D}} | {{SLM_R}} | {{SLM_G}} | No |
| **TCA** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **Yes** |

## 7.10 Training Curves

**Figure 7.1** shows loss vs step. {{TRAINING_CURVES_DESC}}

## 7.11 Summary

- **KV cache:** {{KV_PCT}}% reduction at ρ = {{RATIO}}.
- **Compute:** {{FLOP_PCT}}% attention FLOP reduction.
- **Accuracy:** {{D_DELTA}}% (DistilBERT), {{R_DELTA}}% (RoBERTa), {{G_DELTA}}% (GPT-2).
- **Parameters:** increase by {{D_PARAM_PCT}}% due to LoRA + score_net.

---
