<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT (N=128)

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Path | Dense |
| Time | {{D_TIME}} s |
| VRAM | {{D_VRAM}} MB |

## C.2 DistilBERT (N=512)

| Metric | Baseline | Nyström |
|---|---|---|
| Accuracy | {{D_L512_BASE}}% | {{D_L512_NYS}}% |
| Time | {{D_T512_D}} s | {{D_T512_N}} s |
| Speedup | — | **{{S512}}×** |
| VRAM | {{V512_D}} MB | {{V512_N}} MB |

## C.3 RoBERTa (N=128)

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Path | Dense |
| Time | {{R_TIME}} s |
| VRAM | {{R_VRAM}} MB |

## C.4 GPT-2 (N=128)

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Path | Dense causal |
| Time | {{G_TIME}} s |
| VRAM | {{G_VRAM}} MB |

## C.5 Landmark Ablation

| $m$ | DistilBERT (N=512) | Time | Speedup |
|---|---|---|---|
| 16 | {{D_M16}} | {{D_M16_T}} | {{D_M16_S}}× |
| 32 | {{D_M32}} | {{D_M32_T}} | {{D_M32_S}}× |
| 64 | {{D_M64}} | {{D_M64_T}} | {{D_M64_S}}× |
| 128 | {{D_M128}} | {{D_M128_T}} | {{D_M128_S}}× |

## C.6 Pinv Iterations Ablation

| Iters | Accuracy | Time |
|---|---|---|
| 1 | {{D_I1}} | {{D_I1_T}} |
| 2 | {{D_I2}} | {{D_I2_T}} |
| 3 | {{D_I3}} | {{D_I3_T}} |
| 5 | {{D_I5}} | {{D_I5_T}} |

## C.7 Approximation Error vs N

| $N$ | Error |
|---|---|
| 128 | N/A (dense) |
| 256 | {{E256}} |
| 512 | {{E512}} |
| 1024 | {{E1024}} |
| 2048 | {{E2048}} |

## C.8 Speedup vs N

| $N$ | Dense time | Nyström time | Speedup |
|---|---|---|---|
| 128 | {{D_T128}} | Same | 1.0× |
| 256 | {{D_T256}} | {{N_T256}} | {{S256}}× |
| 512 | {{D_T512}} | {{N_T512}} | {{S512}}× |
| 1024 | {{D_T1024}} | {{N_T1024}} | {{S1024}}× |
| 2048 | {{D_T2048}} | {{N_T2048}} | {{S2048}}× |

## C.9 Memory vs N

| $N$ | Dense VRAM | Nyström VRAM | Reduction |
|---|---|---|---|
| 128 | {{V128_D}} | {{V128_N}} | 1.0× |
| 512 | {{V512_D}} | {{V512_N}} | {{VR512}}× |
| 1024 | {{V1024_D}} | {{V1024_N}} | {{VR1024}}× |
| 2048 | {{V2048_D}} | {{V2048_N}} | {{VR2048}}× |

## C.10 Newton-Schulz Convergence

| Iterations | $\|I - A V_k\|_F$ |
|---|---|
| 0 | {{NS0_ERR}} |
| 1 | {{NS1_ERR}} |
| 2 | {{NS2_ERR}} |
| 3 | {{NS3_ERR}} |
| 5 | {{NS5_ERR}} |

## C.11 Training Loss Curves

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

## C.12 Raw JSON

```json
{
  "distilbert": {
    "n_128": {
      "accuracy_pct": {{D_ACC}},
      "path": "dense",
      "time": {{D_TIME}}
    },
    "n_512": {
      "baseline_accuracy_pct": {{D_L512_BASE}},
      "nystrom_accuracy_pct": {{D_L512_NYS}},
      "speedup": {{S512}}
    }
  },
  "roberta": {...},
  "gpt2": {...}
}
```
```

---