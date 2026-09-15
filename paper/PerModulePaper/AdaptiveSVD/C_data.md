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
| Params (after) | {{D_PARAMS}} |
| Compression ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 (binary) | {{D_F1}} |
| F1 (macro) | {{D_F1_MACRO}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Validation loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Compression time | {{D_TOTAL}} s |
| Compressed layers | {{D_COMP}} |
| Fallback layers | {{D_FALLBACK}} |
| Avg reconstruction error | {{D_AVG_ERR}} |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Compression ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Compression ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Per-Layer Rank Distribution (DistilBERT)

| Layer | Q | K | V | O | Row avg |
|---|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} | {{D_L0_AVG}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} | {{D_L1_AVG}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} | {{D_L2_AVG}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} | {{D_L3_AVG}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} | {{D_L4_AVG}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} | {{D_L5_AVG}} |
| **Col avg** | **{{D_AVG_Q}}** | **{{D_AVG_K}}** | **{{D_AVG_V}}** | **{{D_AVG_O}}** | **{{D_AVG_ALL}}** |

## C.5 Per-Layer Rank Distribution (RoBERTa)

(12 rows for 12 layers)

## C.6 Per-Layer Rank Distribution (GPT-2)

(12 rows for 12 layers, only Q/K/V/O equivalents for c_attn/c_proj)

## C.7 Per-Layer Reconstruction Errors

### DistilBERT

| Layer | Q error | K error | V error | O error |
|---|---|---|---|---|
| 0 | {{D_L0_Q_ERR}} | {{D_L0_K_ERR}} | {{D_L0_V_ERR}} | {{D_L0_O_ERR}} |
| 1 | {{D_L1_Q_ERR}} | {{D_L1_K_ERR}} | {{D_L1_V_ERR}} | {{D_L1_O_ERR}} |
| ... | ... | ... | ... | ... |

## C.8 Energy Threshold Ablation

| τ | DistilBERT acc | Avg rank | Ratio |
|---|---|---|---|
| 0.85 | {{D_T85}} | {{AVG_RANK_85}} | {{D_T85_R}}× |
| 0.90 | {{D_T90}} | {{AVG_RANK_90}} | {{D_T90_R}}× |
| 0.92 | {{D_T92}} | {{AVG_RANK_92}} | {{D_T92_R}}× |
| 0.95 | {{D_T95}} | {{AVG_RANK_95}} | {{D_T95_R}}× |
| 0.99 | {{D_T99}} | {{AVG_RANK_99}} | {{D_T99_R}}× |

## C.9 Residual Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 | Adapter params |
|---|---|---|---|---|
| 0 | {{D_RL0}} | {{R_RL0}} | {{G_RL0}} | 0 |
| 4 | {{D_RL4}} | {{R_RL4}} | {{G_RL4}} | 4·d_in·N_P |
| 8 | {{D_RL8}} | {{R_RL8}} | {{G_RL8}} | ... |
| 16 | {{D_RL16}} | {{R_RL16}} | {{G_RL16}} | ... |
| 32 | {{D_RL32}} | {{R_RL32}} | {{G_RL32}} | ... |
| 64 | {{D_RL64}} | {{R_RL64}} | {{G_RL64}} | ... |

## C.10 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |
| 200 | {{D_S200}} | {{R_S200}} | {{G_S200}} |

## C.11 Rank Cap Ablation

| Cap | DistilBERT | % hit |
|---|---|---|
| 64 | {{D_CAP64}} | {{D_HIT64}}% |
| 128 | {{D_CAP128}} | {{D_HIT128}}% |
| 256 | {{D_CAP256}} | {{D_HIT256}}% |
| 384 | {{D_CAP384}} | {{D_HIT384}}% |

## C.12 Training Loss Curves

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

## C.13 Raw JSON Metrics

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "energy_threshold": {{ENERGY_THRESHOLD}},
    "residual_rank": {{RESIDUAL_RANK}},
    "min_rank_obs": {{MIN_RANK_OBS}},
    "max_rank_obs": {{MAX_RANK_OBS}},
    "layer_ranks": {
      "layer_0_q": {{D_L0_Q}},
      "layer_0_k": {{D_L0_K}},
      "...": "..."
    }
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.14 Reproduction Commands

```bash
# Main
python run_all_4_adaptive_svd.py --models distilbert,roberta,gpt2

# Energy threshold sweep
for tau in 0.85 0.90 0.92 0.95 0.99; do
    python run_all_4_adaptive_svd.py --energy_threshold $tau --output_suffix tau_$tau
done

# Residual rank sweep
for rl in 0 4 8 16 32 64; do
    python run_all_4_adaptive_svd.py --residual_rank $rl --output_suffix rl_$rl
done
```
