<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 Hardware Detection Results

| Check | Result |
|---|---|
| `torch.cuda.is_available()` | True |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Compute capability | (8, 9) |
| `torch.float8_e4m3fn` | {{FP8_DTYPE}} |
| `flash_attn` importable | {{FLASH_AVAILABLE}} |
| FP8 path taken | {{FP8_TAKEN}} |

## C.2 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Tokens/sec | {{D_TPS}} |

## C.3 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Precision | {{R_PREC}} |
| Recall | {{R_REC}} |
| Loss | {{R_LOSS}} |
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |

## C.4 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Precision | {{G_PREC}} |
| Recall | {{G_REC}} |
| Loss | {{G_LOSS}} |
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |

## C.5 Training Loss Curves

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
5     {{R_S5}}
10    {{R_S10}}
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

## C.6 Per-Layer Weight Transfer Verification

### DistilBERT

| Layer | Q diff | K diff | V diff | O diff |
|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| ... | ... | ... | ... | ... |

### RoBERTa

(Same verification; diff should be 0 for Q/K/V, N/A for O due to nested)

### GPT-2

(Same verification; requires Conv1D transpose check)

## C.7 Raw JSON

```json
{
  "distilbert": {
    "accuracy": {{D_ACC}},
    "params_after": {{D_PARAMS}},
    "fp8_path_taken": false,
    "sdpa_backend": "flash",
    "peak_vram_mb": {{D_VRAM}}
  },
  ...
}
```

## C.8 Reproduction Commands

```bash
# Full run
python run_all_4_flash3_fp8.py --models distilbert,roberta,gpt2

# Ablations
python run_all_4_flash3_fp8.py --use_fp8 false --output_suffix nofp8
```

## C.9 Hardware Comparison Table

| GPU | Compute | FP8 Available? |
|---|---|---|
| RTX 4070 Laptop | 8.9 | ❌ (this study) |
| RTX 4090 | 8.9 | ✅ |
| H100 | 9.0 | ✅ |
| A100 | 8.0 | ❌ |
```
```

---
