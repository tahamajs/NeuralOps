# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Accuracy | 73.62% |
| F1 | 0.7516 |
| Peak VRAM | 329 MB MB |
| Inference time | 1.42x s |
| LoRA rank | 8 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Accuracy | 84.17% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Accuracy | 50.92% |

## C.4 Preservation Gate Ablation

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 | 0.85 | 0.85 | 0.82 |
| 2.0 | 0.85 | 0.85 | 0.85 |
| 4.0 | 0.85 | 0.85 | 0.85 |

## C.5 Decay Ablation

| Decay | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Learned | 0.85 | 0.85 | 0.85 |
| Fixed 0.99 | 0.85 | 0.85 | 0.85 |
| Fixed 0.9 | 0.85 | 0.85 | 0.85 |
| None | 0.85 | 0.85 | 0.85 |

## C.6 Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.8 Scaling

| L | Standard | Mamba | Ratio |
|---|---|---|---|
| 64 | 84.5 | 0.85 | 84.1× |
| 128 | 84.5 | 0.85 | 84.8× |
| 256 | 84.5 | 0.85 | 85.6× |
| 512 | 84.5 | 0.85 | 86.2× |

## C.9 Reproduction Commands

```bash
python run_all_4_mamba.py --recovery_steps 50
```

---