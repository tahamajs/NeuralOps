# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Model params | 66,955,010 |
| Trainable (LoRA) | {{D_TRAINABLE}} |
| Trainable % | {{D_TRAIN_PCT}}% |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Training time | {{D_TIME}} s |
| Peak VRAM | {{D_VRAM}} MB |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Model params | 124,647,170 |
| Trainable (LoRA) | {{R_TRAINABLE}} |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Model params | 124,441,344 |
| Trainable (LoRA) | {{G_TRAINABLE}} |
| Accuracy | {{G_ACC}}% |

## C.4 Rank Ablation

| Rank | DistilBERT | Trainable | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 4 | {{D_R4}} | {{D_R4_T}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{D_R8_T}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{D_R16_T}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{D_R32_T}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{D_R64_T}} | {{R_R64}} | {{G_R64}} |

## C.5 Target Ablation

| Target | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q only | {{D_Q}} | {{R_Q}} | {{G_Q}} |
| Q,V | {{D_QV}} | {{R_QV}} | {{G_QV}} |
| Q,K,V | {{D_QKV}} | {{R_QKV}} | {{G_QKV}} |
| All four | {{D_ALL}} | {{R_ALL}} | {{G_ALL}} |
| FFN only | {{D_FFN}} | {{R_FFN}} | {{G_FFN}} |

## C.6 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.7 Compression Recovery

| Compression | Before LoRA | After 50 steps | Recovery |
|---|---|---|---|
| SVD r=384 | {{R384_BEFORE}} | {{R384_AFTER}} | {{R384_REC}} |
| SVD r=128 | {{R128_BEFORE}} | {{R128_AFTER}} | {{R128_REC}} |
| SVD r=64 | {{R64_BEFORE}} | {{R64_AFTER}} | {{R64_REC}} |

## C.8 Reproduction Commands

```bash
python run_all_4_lora_attention.py --lora_rank 16 --lora_alpha 32
```

---