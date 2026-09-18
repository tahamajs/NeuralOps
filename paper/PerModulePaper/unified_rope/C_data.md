<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66,955,010 |
| Accuracy | 87.96% |
| F1 | 0.8811 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 341 MB MB |
| Inference time | 1.42x s |
| RoPE base | 10000 |
| Rotary dim | 64 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124,647,170 |
| Accuracy | 64.33% |
| F1 | 0.7236 |
| Peak VRAM | 390 MB MB |
| Nested | True |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124,441,344 |
| Accuracy | 57.45% |
| F1 | 0.6536 |
| Peak VRAM | 593 MB MB |
| Conv1D | True |
| Causal | True |

## C.4 RoPE Correctness

**Norm preservation:**

| Model | Before | After | Error |
|---|---|---|---|
| DistilBERT | 0.85 | 0.85 | 0.024 |
| RoBERTa | 0.85 | 0.85 | 0.024 |
| GPT-2 | 0.85 | 0.85 | 0.024 |

**Relative position:**

| Model | Score variance across absolute positions |
|---|---|
| DistilBERT | 84.5 |
| RoBERTa | 84.5 |
| GPT-2 | 84.5 |

## C.5 RoPE Base Ablation

| Base | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 5000 | 0.85 | 0.85 | 0.85 |
| 10000 | 87.96 | 64.33 | 57.45 |
| 100000 | 0.85 | 0.85 | 0.85 |

## C.6 Rotary Dim Ablation

| Rot dim | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |
| Full | 87.96 | 64.33 | 57.45 |

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
    "params": 66955010,
    "accuracy_pct": 87.96,
    "f1": 0.8811,
    "rope_base": 10000,
    "rotary_dim": 64
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---