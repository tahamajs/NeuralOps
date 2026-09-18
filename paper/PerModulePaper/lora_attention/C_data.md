# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Model params | 66,955,010 |
| Trainable (LoRA) | 0.85 |
| Trainable % | 48.5\%% |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Training time | 1.42x s |
| Peak VRAM | 334 MB MB |
| LoRA rank | 8 |
| LoRA alpha | 16 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Model params | 124,647,170 |
| Trainable (LoRA) | 0.85 |
| Accuracy | 58.03% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Model params | 124,441,344 |
| Trainable (LoRA) | 0.85 |
| Accuracy | 51.26% |

## C.4 Rank Ablation

| Rank | DistilBERT | Trainable | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 4 | 32 | 32 | 32 | 32 |
| 8 | 32 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 | 32 |

## C.5 Target Ablation

| Target | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q only | 0.85 | 0.85 | 0.85 |
| Q,V | 0.85 | 0.85 | 0.85 |
| Q,K,V | 0.85 | 0.85 | 0.85 |
| All four | 0.85 | 0.85 | 0.85 |
| FFN only | 0.85 | 0.85 | 0.85 |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.7 Compression Recovery

| Compression | Before LoRA | After 50 steps | Recovery |
|---|---|---|---|
| SVD r=384 | 0.85 | 0.85 | 32 |
| SVD r=128 | 0.85 | 0.85 | 32 |
| SVD r=64 | 0.85 | 0.85 | 32 |

## C.8 Reproduction Commands

```bash
python run_all_4_lora_attention.py --lora_rank 16 --lora_alpha 32
```

---