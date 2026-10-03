# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Trainable | 0.85 |
| Accuracy | 90.37% |
| F1 | 0.9050 |
| Order | 2 |
| Temperature | 1.0 |
| Injection success | 0.85/6 |
| Peak VRAM | 339 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | 64.56% |
| Injection success | 0.85/12 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | 50.92% |
| Injection success | 0.85/12 |

## C.4 Order Ablation

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.5 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.6 Reproduction Commands

```bash
python run_all_4_padre.py --order 2 --temperature 1.0
```

---