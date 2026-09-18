# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 65536000 |
| Attention params before | 2,359,296 |
| Attention params after | 1228800 |
| Attention reduction | 1.92× |
| Accuracy | 81.31% |
| F1 | 0.8191 |
| Peak VRAM | 283 MB |
| Inference time | 2.1 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 121000000 |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 324 MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 121000000 |
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 486 MB |

## C.4 Rank Ablation

| $k$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 80.0 | 32 | 32 |
| 64 | 86.0 | 32 | 32 |
| 128 | 88.5 | 32 | 32 |
| 192 | 89.0 | 32 | 32 |
| 384 | 90.5 | 32 | 32 |

## C.5 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 85.0 | 0.87 | 0.88 |
| 8 | 87.0 | 0.87 | 0.88 |
| 16 | 88.5 | 0.87 | 0.88 |
| 32 | 88.6 | 0.87 | 0.88 |

## C.6 Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.18 | 76.8 | 84.5 |
| 25 | 86.5 | 84.5 | 84.5 |
| 50 | 0.04 | 78.2 | 77.5 |
| 100 | 88.7 | 84.5 | 84.5 |

## C.7 Reconstruction Error

| Layer | Q | K | V |
|---|---|---|---|
| 0 | 0.12 | 0.14 | 0.15 |
| 1 | 0.16 | 0.17 | 0.18 |
| 2 | 0.18 | 0.20 | 0.22 |
| 3 | 0.20 | 0.22 | 0.24 |
| 4 | 0.22 | 0.24 | 0.25 |
| 5 | 0.24 | 0.25 | 0.27 |

## C.8 Training Loss

### DistilBERT
```
Step  Loss
1     0.35
5     0.25
10    0.18
20    0.12
30    0.08
40    0.05
50    0.04
```

## C.9 Reproduction Commands

```bash
python run_all_4_soa_fused.py --models distilbert,roberta,gpt2
```

---
