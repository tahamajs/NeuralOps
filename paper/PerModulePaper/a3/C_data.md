<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Full Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Compression time | {{D_COMP_TIME}} s |
| Compressed layers | {{D_COMP}} |
| Fallback layers | {{D_FALLBACK}} |
| Average error | {{D_AVG_ERR}} |

## C.2 RoBERTa — Full Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Compressed layers | {{R_COMP}} |
| Fallback layers | {{R_FALLBACK}} |
| Average error | {{R_AVG_ERR}} |

## C.3 GPT-2 — Full Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Compressed layers | {{G_COMP}} |
| Fallback layers | {{G_FALLBACK}} |
| Average error | {{G_AVG_ERR}} |

## C.4 Per-Layer Reconstruction Errors (DistilBERT)

| Layer | Module | Rank | Error |
|---|---|---|---|
| 0 | q_lin | {{D_L0_Q_R}} | {{D_L0_Q_E}} |
| 0 | k_lin | {{D_L0_K_R}} | {{D_L0_K_E}} |
| 0 | v_lin | {{D_L0_V_R}} | {{D_L0_V_E}} |
| 0 | out_lin | {{D_L0_O_R}} | {{D_L0_O_E}} |
| 0 | lin1 | {{D_L0_L1_R}} | {{D_L0_L1_E}} |
| 0 | lin2 | {{D_L0_L2_R}} | {{D_L0_L2_E}} |
| ... | ... | ... | ... |
| 5 | lin2 | {{D_L5_L2_R}} | {{D_L5_L2_E}} |

## C.5 Per-Layer Reconstruction Errors (RoBERTa)

(Similar table for RoBERTa's 12 layers)

## C.6 Per-Layer Reconstruction Errors (GPT-2)

(Similar table for GPT-2's 12 layers)

## C.7 Training Loss Curves

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

## C.8 LoRA Rank Ablation — Detail

### DistilBERT

| Rank | Acc | Compressed | Fallback | Time |
|---|---|---|---|---|
| 8 | {{D_R8_A}} | {{D_R8_C}} | {{D_R8_F}} | {{D_R8_T}} |
| 16 | {{D_R16_A}} | {{D_R16_C}} | {{D_R16_F}} | {{D_R16_T}} |
| 32 | {{D_R32_A}} | {{D_R32_C}} | {{D_R32_F}} | {{D_R32_T}} |
| 64 | {{D_R64_A}} | {{D_R64_C}} | {{D_R64_F}} | {{D_R64_T}} |

(Similar for RoBERTa and GPT-2)

## C.9 SVD Rank Ablation — Detail

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | {{D_SVD16}} | {{R_SVD16}} | {{G_SVD16}} |
| 32 | {{D_SVD32}} | {{R_SVD32}} | {{G_SVD32}} |
| 64 | {{D_SVD64}} | {{R_SVD64}} | {{G_SVD64}} |
| 128 | {{D_SVD128}} | {{R_SVD128}} | {{G_SVD128}} |
| 256 | {{D_SVD256}} | {{R_SVD256}} | {{G_SVD256}} |
| 384 | {{D_SVD384}} | {{R_SVD384}} | {{G_SVD384}} |

## C.10 Calibration Size Ablation

| N_calib | D Acc | R Acc | G Acc |
|---|---|---|---|
| 100 | {{D_N100}} | {{R_N100}} | {{G_N100}} |
| 500 | {{D_N500}} | {{R_N500}} | {{G_N500}} |
| 1000 | {{D_N1000}} | {{R_N1000}} | {{G_N1000}} |
| 5000 | {{D_N5000}} | {{R_N5000}} | {{G_N5000}} |

## C.11 Raw JSON Metrics

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "compression_ratio": {{D_RATIO}},
    "peak_vram_mb": {{D_VRAM}},
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
python run_all_4_a3.py --models distilbert,roberta,gpt2

# Rank sweep
for rank in 32 64 128 192 384; do
    python run_all_4_a3.py --rank $rank --output_suffix r$rank
done

# Ablations
python run_all_4_a3.py --use_fisher false --output_suffix nofisher
python run_all_4_a3.py --recalibrate_ln false --output_suffix noln
```
