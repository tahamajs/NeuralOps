<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Param Δ | +{{D_PARAM_DELTA}} |
| Param Δ % | +{{D_PARAM_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Tokens/s | {{D_TPS}} |
| Final gate mean | {{D_FINAL_GATE}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Param Δ | +{{R_PARAM_DELTA}} |
| Param Δ % | +{{R_PARAM_PCT}}% |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Param Δ | +{{G_PARAM_DELTA}} |
| Param Δ % | +{{G_PARAM_PCT}}% |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Ablation: Compression Rank

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Inference time |
|---|---|---|---|---|
| 8 | {{D_C8}} | {{R_C8}} | {{G_C8}} | {{D_T8}} |
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} | {{D_T16}} |
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} | {{D_T32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} | {{D_T64}} |

## C.5 Ablation: Window Size

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_W16}} | {{R_W16}} | {{G_W16}} |
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |

## C.6 Ablation: Gate Bias

| bias | σ(bias) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.0 | 0.500 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |
| 1.0 | 0.731 | {{D_GB1}} | {{R_GB1}} | {{G_GB1}} |
| 2.0 | 0.881 | {{D_GB2}} | {{R_GB2}} | {{G_GB2}} |
| 3.0 | 0.953 | {{D_GB3}} | {{R_GB3}} | {{G_GB3}} |

## C.7 Gate Evolution

### DistilBERT

| Step | Gate mean |
|---|---|
| 0 | 0.88 |
| 10 | {{D_G10}} |
| 25 | {{D_G25}} |
| 50 | {{D_G50}} |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "param_delta": {{D_PARAM_DELTA}},
    "accuracy_pct": {{D_ACC}},
    "final_gate_mean": {{D_FINAL_GATE}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_oha.py --models distilbert,roberta,gpt2

# Compression rank sweep
for dc in 8 16 32 64; do
    python run_all_4_oha.py --compression_rank $dc --output_suffix dc$dc
done

# Window sweep
for w in 16 32 64 128; do
    python run_all_4_oha.py --window_size $w --output_suffix w$w
done
```
