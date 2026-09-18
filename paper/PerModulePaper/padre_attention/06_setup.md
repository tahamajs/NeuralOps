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
| `order` $d$ | 2 |
| `temperature` $T$ | 1.0 |
| `max_error` (inject) | 0.024 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch size | 32 |
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
    --order 2 \
    --recovery_steps 50
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