# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:
- Train: 67,349
- Validation: 872
- Test: 1,821
- Majority: 50.92%

## 6.3 MinimalAURORA Hyperparameters

| Hyperparameter | Value |
|---|---|
| `energy` $\varepsilon$ | {{ENERGY}} |
| `rank` (max cap) | {{RANK}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `dropout` | 0.1 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |

## 6.5 Evaluation

- **Quality:** Accuracy, F1, precision, recall
- **Efficiency:** Total params, peak VRAM, inference time
- **Compression:** Parameter ratio, per-layer rank distribution
- **Energy:** Final `output_scale` values

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

RTX 4070 Laptop, 8 GB VRAM.

## 6.8 Reproduction

```bash
python run_all_4_minimal_aurora.py \
    --models distilbert,roberta,gpt2 \
    --energy {{ENERGY}} \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `minimal_aurora` | Our method |
| Companion | `asvd`, `aasvd`, `a3`, `res_svd` |

---
