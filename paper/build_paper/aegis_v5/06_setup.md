# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 AEGIS-V5 Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| `svd_energy` ($\eta$) | 0.95 | SVD rank selection |
| `min_rank` | 4 | Floor for SVD rank |
| `max_rank` | 384 | Ceiling for SVD rank |
| `num_local_heads` | 8 | Local attention heads |
| `num_global_heads` | 4 | Global attention heads |
| `local_window` ($W$) | 64 | Sliding window |
| `rff_features` ($F$) | 128 | RFF dimension |
| `gate_init_bias` | 1.8 | Prior for local (overridden to 2.0) |
| `softpick_temperature` | 1.0 | Attention temperature |

**Critical:** With `max_rank = 384` and `svd_energy = 0.95`, the effective 
SVD rank is often 384, which equals $d/2$ — **no parameter reduction**.

For actual compression, set `max_rank = 128` or lower.

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `recovery_steps` | 50 |
| `batch_size` | 8 |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Optimizer | AdamW |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters
- Peak VRAM
- Inference time
- Effective compute

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_aegis_v5.py \
    --models distilbert,roberta,gpt2 \
    --svd_energy 0.95 \
    --max_rank 128 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `aegis_v5` | Our method |
| `aegis_v4` | Prior version (broken softpick) |
| `aegis_v5_no_split` | No head split (all local) |
| `aegis_v5_no_svd` | No SVD factorization |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.

---