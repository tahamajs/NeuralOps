# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | 800M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 Hyperparameters

### LEKAv3-specific

| Hyperparameter | Value |
|---|---|
| `num_landmarks` (base $L_0$) | 0.0 |
| `local_window_size` | 128 |
| `feature_dim` ($d_\text{feat}$) | 64 |
| Gate init | 2.0 |

### LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `recovery_steps` | 50 |
| `batch_size` | 32 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Precision | FP16 |

## 6.4 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Peak VRAM
- Inference time
- Effective compute (L · L_land + L · W)

## 6.5 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.6 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.7 Reproduction

```bash
python run_all_4_leka_v3.py \
    --models distilbert,roberta,gpt2 \
    --num_landmarks 0.0 \
    --local_window_size 128 \
    --recovery_steps 50
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `leka_v3` | Our method |
| `leka_v3_no_land` | LEKAv3 without landmark branch |
| `leka_v3_no_local` | LEKAv3 without local branch |
| `performer_like` | Fixed-feature linear attention |

## 6.9 Seeds

Single seed (42). Limitation stated in Section 10.

---