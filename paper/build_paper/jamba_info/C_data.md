# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Trainable | 0.85 |
| Accuracy | 89.68% |
| F1 | 0.8984 |
| α final | 0.85 |
| Peak VRAM | 480 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 52.98% |
| F1 | 0.6792 |
| α final | 0.85 |
| Peak VRAM | 527 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 50.80% |
| F1 | 0.6733 |
| α final | 0.85 |
| Peak VRAM | 791 MB MB |

## C.4 Bottleneck Dim Ablation

| $b$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| 256 | 89.68 | 52.98 | 50.80 |
| 512 | 0.85 | 0.85 | 0.85 |

## C.5 No-Bottleneck Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With BN | 89.68 | 52.98 | 50.80 |
| No BN | 0.85 | 0.85 | 0.85 |

## C.6 α Init Ablation

| $w_0$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| -4 | 0.85 | 0.85 | 0.85 |
| -2 | 89.68 | 52.98 | 50.80 |
| 0 | 0.85 | 0.85 | 0.85 |
| +2 | 0.85 | 0.85 | 0.85 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
20    85.7
30    86.2
40    86.4
50    86.4
```

## C.8 Reproduction Commands

```bash
python run_all_4_jamba_info.py --models distilbert,roberta,gpt2
```

---