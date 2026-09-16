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
| Avg bucket size | {{D_BUCKET}} |
| Compression time | {{D_TOTAL}} s |

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

## C.5 LSH Bucket Statistics

### C.5.1 Bucket Size Distribution (DistilBERT, k=4)

| Bucket size | Count | Frequency |
|---|---|---|
| 0 | {{B0_CNT}} | {{B0_FREQ}}% |
| 1-4 | {{B14_CNT}} | {{B14_FREQ}}% |
| 5-16 | {{B516_CNT}} | {{B516_FREQ}}% |
| 17-64 | {{B1764_CNT}} | {{B1764_FREQ}}% |
| >64 | {{B64_CNT}} | {{B64_FREQ}}% |

## C.6 Ablation: CUR Rank

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 8 | {{D_R8}} | {{D_RATIO8}}× | {{R_R8}} | {{R_RATIO8}}× | {{G_R8}} | {{G_RATIO8}}× |
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |

## C.7 Ablation: LSH Rounds

| Rounds | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_LSH1}} | {{R_LSH1}} | {{G_LSH1}} |
| 2 | {{D_LSH2}} | {{R_LSH2}} | {{G_LSH2}} |
| 4 | {{D_LSH4}} | {{R_LSH4}} | {{G_LSH4}} |
| 8 | {{D_LSH8}} | {{R_LSH8}} | {{G_LSH8}} |

## C.8 Ablation: LSH Bits

| Bits | Buckets | DistilBERT | RoBERTa |
|---|---|---|---|
| 2 | 4 | {{D_B2}} | {{R_B2}} |
| 3 | 8 | {{D_B3}} | {{R_B3}} |
| 4 | 16 | {{D_B4}} | {{R_B4}} |
| 6 | 64 | {{D_B6}} | {{R_B6}} |

## C.9 Ablation: Component Contributions

| Config | DistilBERT | Δ |
|---|---|---|
| Full | {{D_FULL}} | — |
| No softpick | {{D_NOSP}} | {{D_NOSP_D}} |
| No LSH | {{D_NOLSH}} | {{D_NOLSH_D}} |
| No CUR | {{D_NOCUR}} | {{D_NOCUR_D}} |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
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
    "avg_bucket_size": {{D_BUCKET}},
    "compression_time_s": {{D_TOTAL}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_phantom.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_phantom.py --cur_rank $r --output_suffix r$r
done
```

---