# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**: 67,349 train / 872 val. Majority baseline 50.92%. 
Max sequence length: 128.

## 6.3 PADRe Hyperparameters

| Hyperparameter | Value |
|---|---|
| `order` $d$ | {{ORDER}} |
| `temperature` $T$ | {{TEMPERATURE}} |
| `max_error` (inject) | {{MAX_ERROR}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| WD | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

- Accuracy, F1
- Peak VRAM, inference time
- Injection success rate
- Output equivalence error

## 6.6 Quality Gate

PASS ≥ 56%, PASS_LOW ≥ 50.92%, FAIL < 50.92%.

## 6.7 Hardware

RTX 4070 Laptop, 8 GB.

## 6.8 Reproduction

```bash
python run_all_4_padre.py \
    --models distilbert,roberta,gpt2 \
    --order {{ORDER}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard softmax attention |
| `padre` | Our method |
| `padre_order1` | Linear polynomial |
| `padre_order2` | Quadratic polynomial |
| `padre_order4` | Quartic polynomial |
| `padre_no_center` | Without max-centering |

---