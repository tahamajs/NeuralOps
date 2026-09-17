# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` F | 64 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `rank` | 32 |
| Preservation gate $w_0$ | 2.0 (σ → 0.88) |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Attention parameters.
- Peak VRAM.
- Inference time at L = 128, 256, 512, 1024, 2048.

### Kernel-specific
- Mean preservation gate α.
- Kernel approximation error (empirical).

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
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_leka_v1.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `leka_v1` | Our method |
| `leka_no_pres_gate` | No preservation gate |
| `leka_no_norm` | No head-wise LayerNorm |
| `leka_no_qr` | Random (non-QR) projection |

## 6.10 Seeds

Single seed (42).

---