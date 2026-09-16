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
| `chunk_size` | {{CHUNK_SIZE}} |
| `log_decay_range` | [{{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}] |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |

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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_decay_linear.py \
    --models distilbert,roberta,gpt2 \
    --chunk_size {{CHUNK_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
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

Single seed ({{SEED}}).

---