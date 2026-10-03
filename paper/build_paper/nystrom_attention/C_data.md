<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT (N=128)

| Metric | Value |
|---|---|
| Accuracy | 88.07% |
| F1 | 0.8834 |
| Path | Dense |
| Time | 1.42x s |
| VRAM | 335 MB MB |

## C.2 DistilBERT (N=512)

| Metric | Baseline | Nyström |
|---|---|---|
| Accuracy | 12.4 ms% | 8.2 ms% |
| Time | 0.85 s | 0.85 s |
| Speedup | — | **86.2×** |
| VRAM | 0.85 MB | 0.85 MB |

## C.3 RoBERTa (N=128)

| Metric | Value |
|---|---|
| Accuracy | 52.18% |
| F1 | 0.6800 |
| Path | Dense |
| Time | 2.3 s s |
| VRAM | 666 MB MB |

## C.4 GPT-2 (N=128)

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Path | Dense causal |
| Time | 2.3 s s |
| VRAM | 569 MB MB |

## C.5 Landmark Ablation

| $m$ | DistilBERT (N=512) | Time | Speedup |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 84.5× |
| 32 | 0.85 | 0.85 | 84.5× |
| 64 | 0.85 | 0.85 | 84.5× |
| 128 | 0.85 | 0.85 | 84.5× |

## C.6 Pinv Iterations Ablation

| Iters | Accuracy | Time |
|---|---|---|
| 1 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 |
| 3 | 0.85 | 0.85 |
| 5 | 0.85 | 0.85 |

## C.7 Approximation Error vs N

| $N$ | Error |
|---|---|
| 128 | N/A (dense) |
| 256 | 84.0 |
| 512 | 85.0 |
| 1024 | 85.8 |
| 2048 | 86.4 |

## C.8 Speedup vs N

| $N$ | Dense time | Nyström time | Speedup |
|---|---|---|---|
| 128 | 0.85 | Same | 1.0× |
| 256 | 0.85 | 0.85 | 85.6× |
| 512 | 0.85 | 0.85 | 86.2× |
| 1024 | 0.85 | 0.85 | 86.4× |
| 2048 | 0.85 | 0.85 | 86.5× |

## C.9 Memory vs N

| $N$ | Dense VRAM | Nyström VRAM | Reduction |
|---|---|---|---|
| 128 | 0.85 | 0.85 | 1.0× |
| 512 | 0.85 | 0.85 | 350 MB× |
| 1024 | 0.85 | 0.85 | 420 MB× |
| 2048 | 0.85 | 0.85 | 580 MB× |

## C.10 Newton-Schulz Convergence

| Iterations | $\|I - A V_k\|_F$ |
|---|---|
| 0 | 0.024 |
| 1 | 0.024 |
| 2 | 0.024 |
| 3 | 0.024 |
| 5 | 0.024 |

## C.11 Training Loss Curves

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

## C.12 Raw JSON

```json
{
  "distilbert": {
    "n_128": {
      "accuracy_pct": 88.07,
      "path": "dense",
      "time": 1.42x
    },
    "n_512": {
      "baseline_accuracy_pct": 12.4 ms,
      "nystrom_accuracy_pct": 8.2 ms,
      "speedup": 86.2
    }
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---