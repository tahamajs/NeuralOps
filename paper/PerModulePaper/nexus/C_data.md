# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Active path | sparse |
| Block size | 16 |
| Top-k | 4 |
| Peak VRAM | 346 MB MB |
| Inference time | 1.42x s |
| Mean gate | 0.85 |
| Mean kept blocks | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | 50.92% |
| Active path | sparse |
| Mean gate | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | 51.61% |
| Active path | **dense (causal)** |

## C.4 Component Ablation

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Full | 0.85 | 0.85 |
| − XAttention | 0.85 | 0.85 |
| − Seer | 0.85 | 0.85 |
| − Stripe (inactive) | — | — |
| − Softpick | 0.85 | 0.85 |
| Dense only | 0.85 | 0.85 |

## C.5 Block Size Ablation

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.7 Reproduction Commands

```bash
python run_all_4_nexus.py --recovery_steps 50
```

---