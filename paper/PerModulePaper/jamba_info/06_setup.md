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
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `bottleneck_dim` | {{BOTTLENECK_DIM}} |
| `mode` | "retrieval" |
| `sparsity` | 0.5 |
| λ (differential) | 0.2 |
| α init | σ(-2) ≈ 0.12 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| WD | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- Total parameters.
- Trainable parameters.
- Peak VRAM.
- Inference time.

### Bottleneck-specific
- Effective rank of SSM state.
- Final α value.

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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_jamba_info.py \
    --models distilbert,roberta,gpt2 \
    --bottleneck_dim {{BOTTLENECK_DIM}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `jamba_info` | Our method |
| `jamba_no_bottleneck` | SSM without bottleneck (ablation) |
| `jamba_attn_only` | Only attention (ablation) |
| `jamba_ssm_only` | Only SSM (ablation) |

## 6.10 Seeds

Single seed ({{SEED}}).

---