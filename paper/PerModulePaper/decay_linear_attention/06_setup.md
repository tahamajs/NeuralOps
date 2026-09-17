# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
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
| `chunk_size` | 32 |
| `log_decay_range` | [-0.5, 0.0] |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| WD | 1e-2 |
| Steps | 50 |
| Batch | 8 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- **State memory** (constant in L).
- **Peak VRAM**.
- Inference time at multiple L.

### Decay Analysis
- Learned decay rates per head.
- Effective horizon $1 / (1 - \lambda)$ per head.

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_decay_linear.py \
    --models distilbert,roberta,gpt2 \
    --chunk_size 32 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `decay_linear` | Our method |
| `decay_linear_no_decay` | Fixed λ = 1 (ablation) |
| `decay_linear_fixed` | Fixed λ per head (no data-dep) |
| CHELA-SLAY | Hybrid |
| Performer | Pure linear |

## 6.10 Seeds

Single seed (42).

---