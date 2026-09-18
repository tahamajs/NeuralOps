<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 63.76% |
| F1 | 0.6708 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Params before | 66,955,010 |
| Params after | 66.4M |
| Peak VRAM | 334 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,647,170 |
| Params after | 124.6M |
| Peak VRAM | 665 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Params before | 124,441,344 |
| Params after | 124.4M |
| Peak VRAM | 569 MB MB |
| Inference time | 2.3 s s |

## C.4 Cluster Count Ablation

| K | DistilBERT | RoBERTa | GPT-2 | Time |
|---|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.5 k-Means Iterations Ablation

| $n_{\text{iters}}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 | 0.85 |

## C.6 Anchor Ablation

| Anchor | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yes | 0.85 | 0.85 | 0.85 |
| No | 0.85 | 0.85 | 0.85 |

## C.7 Self-Token Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Self-token | 84.5 | 84.5 | 84.5 |
| No self-token | 0.85 | 0.85 | 0.85 |

## C.8 Q+K vs Q Ablation

| Feature | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q+K | 0.85 | 0.85 | 0.85 |
| Q only | 0.85 | 0.85 | 0.85 |

## C.9 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
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

## C.10 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 63.76,
    "params_after": 66.4M,
    "num_clusters": 16,
    "peak_vram_mb": 334 MB
  },
  ...
}
```

## C.11 Reproduction

```bash
python run_all_4_routing.py --models distilbert,roberta,gpt2
```

## C.12 Cluster Size Distribution

### DistilBERT, $K=16$

| Cluster | Avg size | Max size | Min size |
|---|---|---|---|
| 0 | [D_C0_AVG] | [D_C0_MAX] | [D_C0_MIN] |
| 1 | [D_C1_AVG] | [D_C1_MAX] | [D_C1_MIN] |
| ... | ... | ... | ... |

---