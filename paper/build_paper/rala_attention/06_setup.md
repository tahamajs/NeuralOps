<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 RALA Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| Rank $r$ (augmentation) | 32 | Per-head rank |
| `alpha` init | 0.0 | Pure linear attention at start |
| Softplus kernel | Yes | — |
| Frozen $U, V$ | 0.85 | If LoRA active |
| LoRA rank | 8 | — |
| LoRA alpha | 16 | — |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

- Accuracy, F1, Precision, Recall
- Total params, Trainable params
- Peak VRAM
- Inference time (median of 20)
- Throughput

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop (8 GB) |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_rala.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed 42.

## 6.10 Baselines

- `full` — unmodified model
- `rala` — our method
- Prior modules: `nha`, `nasc`, `a3`, `flash3_fp8`

---