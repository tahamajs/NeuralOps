<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Compressed layers | {{D_COMP}} |
| Fallback layers | {{D_FALLBACK}} |
| Average error | {{D_AVG_ERR}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | Out |
|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| 2 | {{D_L2_Q}} | {{D_L2_K}} | {{D_L2_V}} | {{D_L2_O}} |
| 3 | {{D_L3_Q}} | {{D_L3_K}} | {{D_L3_V}} | {{D_L3_O}} |
| 4 | {{D_L4_Q}} | {{D_L4_K}} | {{D_L4_V}} | {{D_L4_O}} |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} |

## C.5 Per-Layer Reconstruction Error (RoBERTa)

(12 layers × 4 projections)

## C.6 Per-Layer Reconstruction Error (GPT-2)

(12 layers × 2 projections)

## C.7 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{R_R256}} | {{R_RATIO256}}× | {{G_R256}} | {{G_RATIO256}}× |

## C.8 Covariance Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full cov | {{D_FULLCOV}} | {{R_FULLCOV}} | {{G_FULLCOV}} |
| Diagonal | {{D_DIAG}} | {{R_DIAG}} | {{G_DIAG}} |
| No weighting | {{D_NONE}} | {{R_NONE}} | {{G_NONE}} |

## C.9 EMA β Ablation

| β | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_B5}} | {{R_B5}} | {{G_B5}} |
| 0.7 | {{D_B7}} | {{R_B7}} | {{G_B7}} |
| 0.9 | {{D_B9}} | {{R_B9}} | {{G_B9}} |
| 0.95 | {{D_B95}} | {{R_B95}} | {{G_B95}} |
| 0.99 | {{D_B99}} | {{R_B99}} | {{G_B99}} |

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

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "compressed_layers": {{D_COMP}},
    "fallback_layers": {{D_FALLBACK}},
    "avg_reconstruction_error": {{D_AVG_ERR}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_aasvd.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 32 64 128 256; do
    python run_all_4_aasvd.py --rank $r --output_suffix r$r
done
```
