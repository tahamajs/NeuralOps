# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Energy threshold | {{ENERGY}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Final `output_scale` | {{D_OS}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| Final `output_scale` | {{R_OS}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| Final `output_scale` | {{G_OS}} |

## C.4 Energy Threshold Ablation

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_R}}× | {{R_E90}} | {{G_E90}} |
| 0.95 | {{D_E95}} | {{D_E95_R}}× | {{R_E95}} | {{G_E95}} |
| 0.99 | {{D_E99}} | {{D_E99_R}}× | {{R_E99}} | {{G_E99}} |
| 0.995 | {{D_E995}} | {{D_E995_R}}× | {{R_E995}} | {{G_E995}} |

## C.5 Rank Cap Ablation

| Cap | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_C16}} | {{R_C16}} | {{G_C16}} |
| 32 | {{D_C32}} | {{R_C32}} | {{G_C32}} |
| 64 | {{D_C64}} | {{R_C64}} | {{G_C64}} |
| 128 | {{D_C128}} | {{R_C128}} | {{G_C128}} |

## C.6 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

## C.7 Per-Layer Rank (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| ... | ... | ... | ... | ... |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |

## C.8 Training Loss Curves

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
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "energy": {{ENERGY}},
    "final_output_scale": {{D_OS}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_minimal_aurora.py --energy $e --output_suffix e$e
done
```

---