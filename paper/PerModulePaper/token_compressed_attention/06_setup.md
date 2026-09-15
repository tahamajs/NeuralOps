# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV Heads | Params |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| `compression_ratio` ρ | {{RATIO}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `rank` | {{RANK}} |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

**Trainable parameters per layer:**
- LoRA: `4 · r_L (d_in + d_out)` = `4 · 16 · 1536` = ~98K
- Score net: ~50K
- Total: ~148K per layer

For 6 layers (DistilBERT): ~888K trainable.

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters (baseline vs TCA)
- Peak VRAM (during inference)
- Inference time
- Effective KV cache size
- Attention FLOPs

### Compression
- Compression ratio ρ
- Effective KV length k
- Score-net stats (mean, std)

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
python run_all_4_token_compressed.py \
    --models distilbert,roberta,gpt2 \
    --compression_ratio {{RATIO}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `tca` | Our method |
| `tca_norm_only` | TCA without score_net (ablation) |
| `tca_no_cls` | TCA without [CLS] pinning (ablation) |
| H2O-style | Fixed attention-mass heuristic |

## 6.10 Seeds

Single seed ({{SEED}}). Single-seed limitation stated in Section 10.

---
