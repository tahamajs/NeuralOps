# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

SST-2 (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` F | 64 |
| `landmark_ratio` ρ | 0.25 |
| `min_landmarks` | 64 |
| `local_window_size` W | 128 |
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| Local gate $w_h$ | -2.0 (σ ≈ 0.12) |
| Softpick temperature T | 1.0 |

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
- Inference time at L = 128, 256, 512, 1024.

### Component-specific
- Mean landmark ratio.
- Mean local gate $g_h$.
- Landmark selection distribution.

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
python run_all_4_leka_v2.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 64 \
    --landmark_ratio 0.25 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `leka_v2` | Our method |
| `leka_v2_no_landmark` | Without landmark distillation |
| `leka_v2_no_local` | Without local branch |
| `leka_v2_no_dual` | Single-scale RFF |
| `leka_v2_softmax` | Standard softmax instead of softpick |

## 6.10 Seeds

Single seed (42).

---