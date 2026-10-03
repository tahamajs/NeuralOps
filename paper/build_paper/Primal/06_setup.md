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
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal |

Loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Primal Attention Configuration

| Parameter | Value |
|---|---|
| Landmark count $m$ | 64 |
| Pseudoinverse iterations $K$ | 4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Dropout | 0.1 |

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
- Attention matrix memory (bytes)
- Peak VRAM (MB)
- Inference time (median of 20 runs)

### Approximation Quality
- Frobenius error: $\|A - \tilde{A}\|_F / \|A\|_F$

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
python run_all_4_primal.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --pinv_iters 4 \
    --recovery_steps 50
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| $m$ | $\{16, 32, 64, 128, 256\}$ |
| $K$ (pinv iters) | $\{0, 2, 4, 6, 8\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |
| Recovery steps | $\{10, 25, 50, 100\}$ |

---
