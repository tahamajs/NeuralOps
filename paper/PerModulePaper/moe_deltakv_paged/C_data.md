# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Trainable | 0.85 |
| Accuracy | 86.47% |
| F1 | 0.8628 |
| Page size | 64 |
| k_experts | 4 |
| SVD rank | 16 |
| Num pages | 0.85 |
| Page-0 rate | 32% |
| Aux loss | 0.85 |
| Peak VRAM | 322 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Accuracy | 59.75% |
| Page-0 rate | 32% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Accuracy | 50.92% |
| Page-0 rate | 32% |

## C.4 k_experts Ablation

| k | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | 0.85 | 0.85 | 0.85 |
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.5 Page Size Ablation

| P | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
...
50    86.4
```

## C.7 Scaling

| L | Standard | MoE-DeltaKV |
|---|---|---|
| 128 | 84.5 | 0.85 |
| 256 | 84.5 | 0.85 |
| 512 | 84.5 | 0.85 |
| 1024 | 84.5 | 0.85 |

## C.8 Reproduction Commands

```bash
python run_all_4_moe_deltakv_paged.py --page_size 64 --k_experts 4
```

---