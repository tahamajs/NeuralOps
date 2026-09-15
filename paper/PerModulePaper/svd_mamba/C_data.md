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
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Final gate γ | {{D_FINAL_GATE}} |

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

## C.4 d_state Ablation

| $d_\text{state}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | {{D_DS8}} | {{R_DS8}} | {{G_DS8}} |
| 16 | {{D_DS16}} | {{R_DS16}} | {{G_DS16}} |
| 32 | {{D_DS32}} | {{R_DS32}} | {{G_DS32}} |
| 64 | {{D_DS64}} | {{R_DS64}} | {{G_DS64}} |

## C.5 d_conv Ablation

| $d_\text{conv}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | {{D_DC2}} | {{R_DC2}} | {{G_DC2}} |
| 4 | {{D_DC4}} | {{R_DC4}} | {{G_DC4}} |
| 8 | {{D_DC8}} | {{R_DC8}} | {{G_DC8}} |

## C.6 global_rank Ablation

| $r_g$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_GR4}} | {{R_GR4}} | {{G_GR4}} |
| 8 | {{D_GR8}} | {{R_GR8}} | {{G_GR8}} |
| 16 | {{D_GR16}} | {{R_GR16}} | {{G_GR16}} |

## C.7 Gate Bias Ablation

| bias | $\sigma(b)$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 4.5e-5 | {{D_GB10}} | {{R_GB10}} | {{G_GB10}} |
| -7 | 9.1e-4 | {{D_GB7}} | {{R_GB7}} | {{G_GB7}} |
| -5 | 6.7e-3 | {{D_GB5}} | {{R_GB5}} | {{G_GB5}} |
| 0 | 0.5 | {{D_GB0}} | {{R_GB0}} | {{G_GB0}} |

## C.8 Gate Evolution

### DistilBERT

| Step | γ mean |
|---|---|
| 0 | 0.0009 |
| 10 | {{D_G10}} |
| 25 | {{D_G25}} |
| 50 | {{D_G50}} |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "param_delta": {{D_PARAM_DELTA}},
    "accuracy_pct": {{D_ACC}},
    "final_gate": {{D_FINAL_GATE}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.11 Reproduction Commands

```bash
# Full benchmark
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2

# d_state sweep
for ds in 8 16 32 64; do
    python run_all_4_svd_mamba.py --d_state $ds --output_suffix ds$ds
done

# Gate bias sweep
for gb in -10 -7 -5 0; do
    python run_all_4_svd_mamba.py --gate_bias $gb --output_suffix gb$gb
done
```

---
