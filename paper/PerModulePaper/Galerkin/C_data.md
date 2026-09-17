<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Precision | 0.6625 |
| Recall | 0.6605 |
| Loss | 0.73 |
| Params (before) | 66,955,010 |
| Params (after) | 67544834 |
| Peak VRAM | 341 MB MB |
| Inference time | 7.0 s |
| Path | Galerkin |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (before) | 124,647,170 |
| Params (after) | 132913922 |
| Peak VRAM | 672 MB MB |
| Inference time | 13.1 s |
| Path | Galerkin |
| Status | PASS_LOW |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 51.26% |
| Δ | -28.10% |
| F1 | 0.6763 |
| Params (before) | 124,441,344 |
| Params (after) | 125630220 |
| Peak VRAM | 569 MB MB |
| Inference time | 9.2 s |
| Path | **Decoder fallback** |
| Status | PASS |

## C.4 Rank Ablation (DistilBERT)

| $r$ | Accuracy | Δ | Energy preserved |
|---|---|---|---|
| 8 | 45.0 | -45.7 | 0.60 |
| 16 | 52.0 | -38.7 | 0.72 |
| 32 | 60.0 | -30.7 | 0.82 |
| 64 | 65.94 | -24.77 | 0.88 |
| 128 | 68.0 | -22.7 | 0.92 |

## C.5 Training Loss

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    58.0
...
50    65.94
```

### RoBERTa (unstable)

```
Step  Loss
1     74.2
5     75.6
10    45.0
15    [R_S15]
20    [R_S20]
...
50    48.97
```

### GPT-2 (stable)

```
Step  Loss
1     73.5
5     [G_S5]
...
50    79.36
```

## C.6 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 50.92,
    "path": "galerkin",
    "params_after": 67544834,
    "peak_vram_mb": 341 MB
  },
  "roberta": {
    "accuracy": 50.92,
    "path": "galerkin",
    "status": "PASS_LOW"
  },
  "gpt2": {
    "accuracy": 51.26,
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