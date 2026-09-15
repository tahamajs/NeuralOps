<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS_FULL}} |
| Parameter change | +{{D_PARAM_DELTA}} (+{{D_PARAM_DELTA_PCT}}%) |
| Accuracy | {{D_ACC}}% |
| F1 (binary) | {{D_F1}} |
| F1 (macro) | {{D_F1_MACRO}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Attention sparsity | {{D_SPARSITY}}% |
| Effective support | {{D_SUPPORT}} |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS_FULL}} |
| Parameter change | +{{R_PARAM_DELTA}} (+{{R_PARAM_DELTA_PCT}}%) |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS_FULL}} |
| Parameter change | +{{G_PARAM_DELTA}} (+{{G_PARAM_DELTA_PCT}}%) |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Attention Sparsity by Layer (DistilBERT, α = 1.5)

| Layer | Sparsity | Effective support |
|---|---|---|
| 0 | {{D_L0_SP}}% | {{D_L0_SUP}} |
| 1 | {{D_L1_SP}}% | {{D_L1_SUP}} |
| 2 | {{D_L2_SP}}% | {{D_L2_SUP}} |
| 3 | {{D_L3_SP}}% | {{D_L3_SUP}} |
| 4 | {{D_L4_SP}}% | {{D_L4_SUP}} |
| 5 | {{D_L5_SP}}% | {{D_L5_SUP}} |

## C.5 α Ablation — Full Data

| α | D acc | Sparsity | R acc | Sparsity | G acc | Sparsity |
|---|---|---|---|---|---|---|
| 1.0 | {{D_A1}} | 0% | {{R_A1}} | 0% | {{G_A1}} | 0% |
| 1.2 | {{D_A12}} | {{SP12}}% | {{R_A12}} | {{R_SP12}}% | {{G_A12}} | {{G_SP12}}% |
| 1.5 | {{D_ACC}} | {{SP15}}% | {{R_ACC}} | {{R_SP15}}% | {{G_ACC}} | {{G_SP15}}% |
| 1.8 | {{D_A18}} | {{SP18}}% | {{R_A18}} | {{R_SP18}}% | {{G_A18}} | {{G_SP18}}% |
| 2.0 | {{D_A2}} | {{SP20}}% | {{R_A2}} | {{R_SP20}}% | {{G_A2}} | {{G_SP20}}% |

## C.6 Splash Radius Ablation (GPT-2)

| W | Acc | Effective context |
|---|---|---|
| 8 | {{G_W8}} | 8 |
| 16 | {{G_W16}} | 16 |
| 32 | {{G_ACC}} | 32 |
| 64 | {{G_W64}} | 64 |
| 128 | {{G_W128}} | 128 (full) |

## C.7 Sink Ablation

| Model | With sink | No sink | Δ |
|---|---|---|---|
| DistilBERT | {{D_ACC}} | {{D_NOSINK}} | {{D_SINK_DELTA}} |
| RoBERTa | {{R_ACC}} | {{R_NOSINK}} | {{R_SINK_DELTA}} |
| GPT-2 | {{G_ACC}} | {{G_NOSINK}} | {{G_SINK_DELTA}} |

## C.8 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} |
| 8 | {{D_RL8}} | {{R_RL8}} | {{G_RL8}} |
| 16 | {{D_RL16}} | {{R_RL16}} | {{G_RL16}} |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} |

## C.9 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

(Similar for RoBERTa, GPT-2)

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "parameter_change": "{{D_PARAM_DELTA_PCT}}%",
    "accuracy_pct": {{D_ACC}},
    "sparsity_pct": {{D_SPARSITY}},
    "alpha": {{ALPHA}},
    "splash_radius": {{SPLASH_RADIUS}},
    "lora_rank": {{LORA_RANK}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Main
python run_all_4_adasplash.py --models distilbert,roberta,gpt2

# Alpha sweep
for alpha in 1.0 1.2 1.5 1.8 2.0; do
    python run_all_4_adasplash.py --alpha $alpha --output_suffix a$alpha
done

# Splash radius sweep
for W in 8 16 32 64 128; do
    python run_all_4_adasplash.py --splash_radius $W --output_suffix W$W
done
```
