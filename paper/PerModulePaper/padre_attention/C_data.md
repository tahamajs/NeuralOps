# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params | 66.95M |
| Trainable | {{D_TRAINABLE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Order | {{ORDER}} |
| Temperature | {{TEMPERATURE}} |
| Injection success | {{D_INJ}}/6 |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params | 124.6M |
| Accuracy | {{R_ACC}}% |
| Injection success | {{R_INJ}}/12 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params | 124.4M |
| Accuracy | {{G_ACC}}% |
| Injection success | {{G_INJ}}/12 |

## C.4 Order Ablation

| Order | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_O1}} | {{R_O1}} | {{G_O1}} |
| 2 | {{D_O2}} | {{R_O2}} | {{G_O2}} |
| 4 | {{D_O4}} | {{R_O4}} | {{G_O4}} |
| 8 | {{D_O8}} | {{R_O8}} | {{G_O8}} |

## C.5 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
...
50    {{D_S50}}
```

## C.6 Reproduction Commands

```bash
python run_all_4_padre.py --order 2 --temperature 1.0
```
```

---