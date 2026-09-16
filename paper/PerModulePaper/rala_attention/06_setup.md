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
| Rank $r$ (augmentation) | {{RALA_RANK}} | Per-head rank |
| `alpha` init | 0.0 | Pure linear attention at start |
| Softplus kernel | Yes | — |
| Frozen $U, V$ | {{U_V_FROZEN}} | If LoRA active |
| LoRA rank | {{LORA_RANK}} | — |
| LoRA alpha | {{LORA_ALPHA}} | — |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_rala.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Seeds

Single seed {{SEED}}.

## 6.10 Baselines

- `full` — unmodified model
- `rala` — our method
- Prior modules: `nha`, `nasc`, `a3`, `flash3_fp8`

---