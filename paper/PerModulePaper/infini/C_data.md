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
| Memory params | {{D_MEM_PARAMS}} |
| LoRA params | {{D_LORA_PARAMS}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Final β | {{D_FINAL_BETA}} |
| Final σ(β) | {{D_FINAL_SIGMA}} |

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

## C.4 Ablation: Compression Dimension

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} | +{{M32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} | +{{M64}} |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} | +{{M128}} |
| 256 | {{D_C256}} | {{R_C256}} | {{G_C256}} | +{{M256}} |

## C.5 Ablation: Window Size

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | {{D_W32}} | {{R_W32}} | {{G_W32}} |
| 64 | {{D_W64}} | {{R_W64}} | {{G_W64}} |
| 128 | {{D_W128}} | {{R_W128}} | {{G_W128}} |
| 256 | {{D_W256}} | {{R_W256}} | {{G_W256}} |

## C.6 Ablation: β Init

| β init | σ(β) init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 0.00005 | {{D_B10}} | {{R_B10}} | {{G_B10}} |
| -5 | 0.0067 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| -2 | 0.12 | {{D_B2}} | {{R_B2}} | {{G_B2}} |
| 0 | 0.50 | {{D_B0}} | {{R_B0}} | {{G_B0}} |

## C.7 β Evolution During Training

### DistilBERT

| Step | β | σ(β) |
|---|---|---|
| 0 | -5.0 | 0.0067 |
| 10 | {{D_B10_B}} | {{D_B10_S}} |
| 25 | {{D_B25_B}} | {{D_B25_S}} |
| 50 | {{D_B50_B}} | {{D_B50_S}} |

### RoBERTa

| Step | β | σ(β) |
|---|---|---|
| 0 | -5.0 | 0.0067 |
| 10 | {{R_B10_B}} | {{R_B10_S}} |
| 25 | {{R_B25_B}} | {{R_B25_S}} |
| 50 | {{R_B50_B}} | {{R_B50_S}} |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
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
    "final_beta": {{D_FINAL_BETA}},
    "final_sigma": {{D_FINAL_SIGMA}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_infini.py --models distilbert,roberta,gpt2

# Compression dim sweep
for dc in 32 64 128 256; do
    python run_all_4_infini.py --compress_dim $dc --output_suffix dc$dc
done

# Window sweep
for w in 32 64 128 256; do
    python run_all_4_infini.py --local_window $w --output_suffix w$w
done
```

---