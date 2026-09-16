# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Active path | sparse |
| Block size | 16 |
| Top-k | 4 |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Mean gate | {{D_MEANGATE}} |
| Mean kept blocks | {{D_MEANKEEP}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | {{R_ACC}}% |
| Active path | sparse |
| Mean gate | {{R_MEANGATE}} |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | {{G_ACC}}% |
| Active path | **dense (causal)** |

## C.4 Component Ablation

| Config | DistilBERT | RoBERTa |
|---|---|---|
| Full | {{D_FULL}} | {{R_FULL}} |
| − XAttention | {{D_NOXA}} | {{R_NOXA}} |
| − Seer | {{D_NOSEER}} | {{R_NOSEER}} |
| − Stripe (inactive) | — | — |
| − Softpick | {{D_NOSOFTPICK}} | {{R_NOSOFTPICK}} |
| Dense only | {{D_DENSE}} | {{R_DENSE}} |

## C.5 Block Size Ablation

| $B_s$ | DistilBERT | RoBERTa |
|---|---|---|
| 8 | {{D_B8}} | {{R_B8}} |
| 16 | {{D_B16}} | {{R_B16}} |
| 32 | {{D_B32}} | {{R_B32}} |
| 64 | {{D_B64}} | {{R_B64}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
...
50    {{D_S50}}
```

## C.7 Reproduction Commands

```bash
python run_all_4_nexus.py --recovery_steps 50
```

---