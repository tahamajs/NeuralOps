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

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `feature_dim` | 128 |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `gate_init_bias` | 5.0 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| WD | 1e-2 |
| Steps | 50 |
| Batch | 4 |
| Precision | FP16 |

**Trainable:** LoRA + gate ($\Omega$ is frozen buffer).

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Trainable parameters.
- Peak VRAM (with cumsum memory footprint).
- Inference time.

### Stream Analysis
- Gate statistics per layer.
- Dense vs linear contribution.

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
python run_all_4_chela_slay.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim 128 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `chela_slay` | Our method |
| `chela_slay_dense_only` | Only dense stream (ablation) |
| `chela_slay_linear_only` | Only linear stream (ablation) |
| Performer | Pure linear attention |

## 6.10 Seeds

Single seed (42).

---