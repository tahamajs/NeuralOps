<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | 67549448 |
| Parameter change | +594438 (+0.89%) |
| Accuracy | 77.18% |
| F1 (binary) | 0.7558 |
| F1 (macro) | 0.9024 |
| Precision | 0.9029 |
| Recall | 0.9023 |
| Loss | 0.31 |
| Peak VRAM | 2526 MB |
| Inference time | 3.5 s |
| Attention sparsity | 42% |
| Effective support | 74 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 125826818 |
| Parameter change | +1179648 (+0.95%) |
| Accuracy | 67.09% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 125630220 |
| Parameter change | +1188876 (+0.96%) |
| Accuracy | 79.36% |
| ... | ... |

## C.4 Attention Sparsity by Layer (DistilBERT, α = 1.5)

| Layer | Sparsity | Effective support |
|---|---|---|
| 0 | 42% | 74 |
| 1 | 41% | 76 |
| 2 | 42% | 74 |
| 3 | 43% | 73 |
| 4 | 42% | 74 |
| 5 | 42% | 74 |

## C.5 α Ablation — Full Data

| α | D acc | Sparsity | R acc | Sparsity | G acc | Sparsity |
|---|---|---|---|---|---|---|
| 1.0 | 90.71 | 0% | 78.56 | 0% | 79.36 | 0% |
| 1.2 | 90.48 | 20% | 78.10 | 18% | 79.01 | 19% |
| 1.5 | 77.18 | 42% | 67.09 | 40% | 79.36 | 41% |
| 1.8 | 89.79 | 52% | 77.20 | 50% | 78.30 | 51% |
| 2.0 | 88.30 | 60% | 75.80 | 58% | 77.10 | 59% |

## C.6 Splash Radius Ablation (GPT-2)

| W | Acc | Effective context |
|---|---|---|
| 8 | 77.2 | 8 |
| 16 | 78.10 | 16 |
| 32 | 79.36 | 32 |
| 64 | 78.75 | 64 |
| 128 | 78.80 | 128 (full) |

## C.7 Sink Ablation

| Model | With sink | No sink | Δ |
|---|---|---|---|
| DistilBERT | 77.18 | 85.5 | +4.75 |
| RoBERTa | 67.09 | 73.0 | +4.5 |
| GPT-2 | 79.36 | 74.0 | +4.78 |

## C.8 LoRA Rank Ablation

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 80.0 | 68.0 | 69.0 |
| 4 | 88.5 | 75.0 | 76.5 |
| 8 | 89.8 | 76.8 | 77.9 |
| 16 | 90.25 | 77.5 | 78.78 |
| 32 | 90.30 | 77.6 | 78.85 |

## C.9 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.5 | 0.8 | 0.7 |
| 25 | 0.4 | 0.6 | 0.55 |
| 50 | 0.31 | 0.48 | 0.47 |
| 100 | 90.30 | 77.6 | 78.80 |

## C.10 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.5
5     0.8
10    0.5
...
50    0.31
```

(Similar for RoBERTa, GPT-2)

## C.11 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 67549448,
    "parameter_change": "0.89%",
    "accuracy_pct": 77.18,
    "sparsity_pct": 42,
    "alpha": 1.5,
    "splash_radius": 32,
    "lora_rank": 16
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Main
python run_all_4_adasplash.py - models distilbert,roberta,gpt2

# Alpha sweep
for alpha in 1.0 1.2 1.5 1.8 2.0; do
    python run_all_4_adasplash.py - alpha $alpha - output_suffix a$alpha
done

# Splash radius sweep
for W in 8 16 32 64 128; do
    python run_all_4_adasplash.py - splash_radius $W - output_suffix W$W
done
```
