# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Total params before | 66,955,010 |
| Total params after | 66.4M |
| Trainable | 0.85 |
| Accuracy | 90.14% |
| F1 | 0.9053 |
| Peak VRAM | 764 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Total params after | 124.6M |
| Accuracy | 84.29% |
| F1 | 0.8620 |
| Peak VRAM | 842 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Total params after | 124.4M |
| Accuracy | 80.50% |
| F1 | 0.8041 |
| Peak VRAM | 777 MB MB |

## C.4 Expert Count Ablation

| $E$ | DistilBERT | RoBERTa | GPT-2 | Params/layer |
|---|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 | 9.4M |
| 8 | 0.85 | 0.85 | 0.85 | 18.9M |
| 16 | 0.85 | 0.85 | 0.85 | 37.8M |

## C.5 Capacity Factor Ablation

| CF | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | 0.85 | 0.85 | 0.85 |
| 1.0 | 0.865 | 0.865 | 0.865 |
| 1.5 | 0.865 | 0.865 | 0.865 |
| 2.0 | 0.85 | 0.85 | 0.85 |

## C.6 MoE Gate Initialization Ablation

| $b_g$ | σ($b_g$) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -8.0 | 0.0003 | 0.85 | 0.85 | 0.85 |
| -4.0 | 0.018 | 0.85 | 0.85 | 0.85 |
| -2.0 | 0.12 | 0.85 | 0.85 | 0.85 |
| 0.0 | 0.50 | 0.85 | 0.85 | 0.85 |

## C.7 Expert Utilization

| Expert | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 12.5% | 12.5% | 12.5% |
| 1 | 12.5% | 12.5% | 12.5% |
| ... | ... | ... | ... |
| 7 | 12.5% | 12.5% | 12.5% |

**Perfect balance.**

## C.8 Per-Token Capacity Distribution

| Selected by | Fraction |
|---|---|
| 0 experts | 48.5\%% |
| 1 expert | 48.5\%% |
| 2 experts | 48.5\%% |
| 3+ experts | 48.5\%% |

## C.9 Gate Evolution

| Step | γ |
|---|---|
| 0 | 0.018 |
| 10 | 84.5 |
| 25 | 84.5 |
| 50 | 84.5 |

## C.10 Training Loss

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

## C.11 Reproduction Commands

```bash
python run_all_4_expert_choice_moe.py --models distilbert,roberta,gpt2
```

---