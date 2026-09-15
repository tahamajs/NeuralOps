<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Path | Galerkin |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Peak VRAM | {{R_VRAM}} MB |
| Inference time | {{R_TIME}} s |
| Path | Galerkin |
| Status | {{R_STATUS}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Peak VRAM | {{G_VRAM}} MB |
| Inference time | {{G_TIME}} s |
| Path | **Decoder fallback** |
| Status | {{G_STATUS}} |

## C.4 Rank Ablation (DistilBERT)

| $r$ | Accuracy | Δ | Energy preserved |
|---|---|---|---|
| 8 | {{D_R8}} | {{D_R8_D}} | {{D_R8_E}} |
| 16 | {{D_R16}} | {{D_R16_D}} | {{D_R16_E}} |
| 32 | {{D_R32}} | {{D_R32_D}} | {{D_R32_E}} |
| 64 | {{D_R64}} | {{D_R64_D}} | {{D_R64_E}} |
| 128 | {{D_R128}} | {{D_R128_D}} | {{D_R128_E}} |

## C.5 Training Loss

### DistilBERT

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

### RoBERTa (unstable)

```
Step  Loss
1     {{R_S1}}
5     {{R_S5}}
10    {{R_S10}}
15    {{R_S15}}
20    {{R_S20}}
...
50    {{R_S50}}
```

### GPT-2 (stable)

```
Step  Loss
1     {{G_S1}}
5     {{G_S5}}
...
50    {{G_S50}}
```

## C.6 Raw JSON

```json
{
  "distilbert": {
    "accuracy": {{D_ACC}},
    "path": "galerkin",
    "params_after": {{D_PARAMS}},
    "peak_vram_mb": {{D_VRAM}}
  },
  "roberta": {
    "accuracy": {{R_ACC}},
    "path": "galerkin",
    "status": "{{R_STATUS}}"
  },
  "gpt2": {
    "accuracy": {{G_ACC}},
    "path": "decoder_fallback",
    "note": "no-op"
  }
}
```

## C.7 Reproduction Commands

```bash
# Main benchmark
python run_all_4_galerkin.py --models distilbert,roberta,gpt2

# Rank ablation
for rank in 8 16 32 64 128; do
    python run_all_4_galerkin.py --rank $rank --output_suffix r$rank
done

# Recovery ablation
for steps in 10 25 50 100; do
    python run_all_4_galerkin.py --recovery_steps $steps --output_suffix s$steps
done
```

---
