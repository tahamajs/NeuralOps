<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | H | d_h | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 12 | 64 | 66.9M |
| RoBERTa-base | 12 | 12 | 64 | 124.6M |
| GPT-2 | 12 | 12 | 64 | 124.4M |

Loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Squeezing Heads Configuration

| Parameter | Value |
|---|---|
| LoRA rank $r_L$ | 16 |
| LoRA alpha $\alpha$ | 32 |
| Head-dim reduction $s$ | 1.0× |
| Target head_dim $d_h'$ | 64 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Trainable params
- Inference time
- Peak VRAM

### Distillation Analysis
- Accuracy vs head-dim reduction
- LoRA rank vs recovery
- Per-layer distillation quality

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
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
python run_all_4_squeezing_heads.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank 16 \
    --head_dim_reduction 1.0 \
    --recovery_steps 50
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| LoRA rank $r_L$ | $\{4, 8, 16, 32\}$ |
| Head-dim reduction | $\{1.0, 1.5, 2.0, 2.5\}$ |
| Recovery steps | $\{10, 25, 50, 100\}$ |
| GQA pooling | $\{\text{mean}, \text{first}\}$ |

---
