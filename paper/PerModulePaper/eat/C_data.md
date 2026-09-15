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
| Compressed layers | {{D_COMP}} |

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

## C.4 LoRA Rank Ablation

### DistilBERT

| Rank | Accuracy | Trainable | Param Δ |
|---|---|---|---|
| 4 | {{D_R4}} | {{D_R4_T}} | +{{D_R4_P}} |
| 8 | {{D_R8}} | {{D_R8_T}} | +{{D_R8_P}} |
| 16 | {{D_R16}} | {{D_R16_T}} | +{{D_R16_P}} |
| 32 | {{D_R32}} | {{D_R32_T}} | +{{D_R32_P}} |
| 64 | {{D_R64}} | {{D_R64_T}} | +{{D_R64_P}} |

(Similar for RoBERTa, GPT-2)

## C.5 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## C.6 Pruning Ablation

| Prune ratio | DistilBERT | Time | RoBERTa | Time |
|---|---|---|---|---|
| 1.0 | {{D_P100}} | {{D_T100}} | {{R_P100}} | {{R_T100}} |
| 0.9 | {{D_P90}} | {{D_T90}} | {{R_P90}} | {{R_T90}} |
| 0.7 | {{D_P70}} | {{D_T70}} | {{R_P70}} | {{R_T70}} |
| 0.5 | {{D_P50}} | {{D_T50}} | {{R_P50}} | {{R_T50}} |
| 0.3 | {{D_P30}} | {{D_T30}} | {{R_P30}} | {{R_T30}} |

## C.7 Training Loss Curves

### DistilBERT

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
15    {{D_S15}}
20    {{D_S20}}
25    {{D_S25}}
30    {{D_S30}}
35    {{D_S35}}
40    {{D_S40}}
45    {{D_S45}}
50    {{D_S50}}
```

### RoBERTa

```
Step  Loss
1     {{R_S1}}
...
50    {{R_S50}}
```

### GPT-2

```
Step  Loss
1     {{G_S1}}
...
50    {{G_S50}}
```

## C.8 Per-Layer Attention Diagnostics

### DistilBERT

| Layer | q norm | k norm | v norm | out norm |
|---|---|---|---|---|
| 0 | {{D_L0_QN}} | {{D_L0_KN}} | {{D_L0_VN}} | {{D_L0_ON}} |
| 1 | {{D_L1_QN}} | {{D_L1_KN}} | {{D_L1_VN}} | {{D_L1_ON}} |
| ... | ... | ... | ... | ... |
| 5 | {{D_L5_QN}} | {{D_L5_KN}} | {{D_L5_VN}} | {{D_L5_ON}} |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "param_delta": {{D_PARAM_DELTA}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "compressed_layers": {{D_COMP}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_eat.py --models distilbert,roberta,gpt2

# LoRA rank sweep
for r in 4 8 16 32 64; do
    python run_all_4_eat.py --lora_rank $r --output_suffix lora$r
done

# Pruning sweep
for ratio in 1.0 0.9 0.7 0.5 0.3; do
    python run_all_4_eat.py --prune_ratio $ratio --output_suffix prune$ratio
done
```

---
