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
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Avg reconstruction error | {{D_AVG_ERR}} |
| Avg rank | {{D_AVG_RANK}} |
| Compression time | {{D_COMPRESS_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Per-Layer Reconstruction Error (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q_E}} | {{D_L0_K_E}} | {{D_L0_V_E}} | {{D_L0_O_E}} |
| 1 | {{D_L1_Q_E}} | {{D_L1_K_E}} | {{D_L1_V_E}} | {{D_L1_O_E}} |
| 2 | {{D_L2_Q_E}} | {{D_L2_K_E}} | {{D_L2_V_E}} | {{D_L2_O_E}} |
| 3 | {{D_L3_Q_E}} | {{D_L3_K_E}} | {{D_L3_V_E}} | {{D_L3_O_E}} |
| 4 | {{D_L4_Q_E}} | {{D_L4_K_E}} | {{D_L4_V_E}} | {{D_L4_O_E}} |
| 5 | {{D_L5_Q_E}} | {{D_L5_K_E}} | {{D_L5_V_E}} | {{D_L5_O_E}} |

## C.5 Per-Layer Selected Rank (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| ... | ... | ... | ... | ... |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |

## C.6 Rank Ablation

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |

## C.7 Energy Threshold Ablation

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_R}}× | {{R_E90}} | {{G_E90}} |
| 0.95 | {{D_E95}} | {{D_E95_R}}× | {{R_E95}} | {{G_E95}} |
| 0.99 | {{D_E99}} | {{D_E99_R}}× | {{R_E99}} | {{G_E99}} |
| 0.995 | {{D_E995}} | {{D_E995_R}}× | {{R_E995}} | {{G_E995}} |

## C.8 Dynamic vs Fixed Rank

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dynamic | {{D_DYN}} | {{R_DYN}} | {{G_DYN}} |
| Fixed | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |

## C.9 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

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

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "avg_rank": {{D_AVG_RANK}},
    "avg_reconstruction_error": {{D_AVG_ERR}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_asvd.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 16 32 64 128; do
    python run_all_4_asvd.py --rank $r --output_suffix r$r
done

# Energy sweep
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_asvd.py --energy_threshold $e --output_suffix e$e
done
```
```

---