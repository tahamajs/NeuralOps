<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | H | d_h | Params | Attention |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 12 | 64 | 66.9M | Flat |
| RoBERTa-base | 12 | 12 | 64 | 124.6M | Nested |
| GPT-2 | 12 | 12 | 64 | 124.4M | Causal |

Loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 0.85.

## 6.3 PTP Configuration

| Parameter | Value |
|---|---|
| Keep ratio $\rho$ | 0.5 |
| Score network hidden | 64 |
| LoRA rank | 8 |
| LoRA alpha | 16 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| Weight decay | 0.01 |
| Steps | 50 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Peak VRAM
- Inference time
- Pruned tokens per layer

### Score Analysis
- Learned score distribution
- Token importance ranking
- Correlation with TF-IDF

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
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_ptp.py \
    --models distilbert,roberta,gpt2 \
    --keep_ratio 0.5 \
    --recovery_steps 50
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Keep ratio $\rho$ | $\{0.5, 0.7, 0.8, 0.9, 0.95\}$ |
| Score network hidden | $\{32, 64, 128\}$ |
| [CLS] preservation | $\{\text{on}, \text{off}\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |

---