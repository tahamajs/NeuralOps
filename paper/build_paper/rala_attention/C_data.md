<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 60.09% |
| F1 | 0.6306 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Peak VRAM | 335 MB MB |
| Inference time | 1.42x s |
| Tokens/sec | 0.85 |
| Alpha final | 0.01 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.80% |
| F1 | 0.6305 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Peak VRAM | 666 MB MB |
| Inference time | 2.3 s s |
| Alpha final | 0.85 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Peak VRAM | 572 MB MB |
| Inference time | 2.3 s s |
| Alpha final | 0.85 |

## C.4 Alpha Trajectory

### DistilBERT

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

### RoBERTa

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

### GPT-2

| Step | Alpha |
|---|---|
| 0 | 0.0000 |
| 50 | 0.85 |

## C.5 Rank Ablation (DistilBERT)

| Rank | Acc | Extra params | Time |
|---|---|---|---|
| 1 | 32 | 6,150 | 32 |
| 2 | 32 | 12,294 | 32 |
| 4 | 32 | 24,582 | 32 |
| 8 | 32 | 49,158 | 32 |
| 16 | 32 | 98,310 | 32 |

## C.6 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| softplus | 84.5 | 84.5 | 84.5 |
| elu+1 | 0.85 | 0.85 | 0.85 |
| ReLU | 32 | 32 | 32 |

## C.7 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

### RoBERTa

```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2

```
Step  Loss
1     73.5
...
50    77.5
```

## C.8 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 60.09,
    "params_after": 66.4M,
    "alpha_final": 0.01,
    "peak_vram_mb": 335 MB
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.9 Per-Layer Alpha

| Layer | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | [D_L0_A] | [R_L0_A] | [G_L0_A] |
| 1 | [D_L1_A] | [R_L1_A] | [G_L1_A] |
| ... | ... | ... | ... |

---