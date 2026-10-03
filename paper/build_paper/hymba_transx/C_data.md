# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Params change | +48.5\%% |
| Trainable | 0.85 |
| Accuracy | 49.08% |
| F1 | 86.2 |
| α final | 0.85 |
| Peak VRAM | 5947 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params after | 124.6M |
| Accuracy | 51.15% |
| F1 | 0.6758 |
| α final | 0.85 |
| Peak VRAM | 534 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params after | 124.4M |
| Accuracy | 57.57% |
| F1 | 0.6859 |
| α final | 0.85 |
| Peak VRAM | 804 MB MB |

## C.4 Ablation: Differential

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Standard | 0.85 | 0.85 | 0.85 |
| Diff λ=0.2 | 49.08 | 51.15 | 57.57 |
| Diff λ=0.8 | 0.85 | 0.85 | 0.85 |

## C.5 Ablation: Fusion

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| α=1 (attn only) | 0.85 | 0.85 | 0.85 |
| α=0 (ssm only) | 84.5 | 84.5 | 84.5 |
| α=0.5 (fixed) | 0.85 | 0.85 | 0.85 |
| α learned | 49.08 | 51.15 | 57.57 |

## C.6 LoRA Rank

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## C.7 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.8 α Per Layer (DistilBERT)

| Layer | α |
|---|---|
| 0 | 0.85 |
| 1 | 0.85 |
| 2 | 0.85 |
| 3 | 0.85 |
| 4 | 0.85 |
| 5 | 0.85 |

## C.9 Training Loss

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

## C.10 Reproduction Commands

```bash
python run_all_4_hymba_transx.py --models distilbert,roberta,gpt2
```

---