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

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 ASA-Span Configuration

| Parameter | Value |
|---|---|
| Max span $S_\text{max}$ | 128 |
| Ramp size $R$ | 32 |
| Initial logit | 4.0 |
| L1 span penalty $\lambda$ | 0.01 |
| LoRA rank | 16 |
| LoRA alpha | 32 |

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
- Learned span per head
- Attention memory
- Inference time

### Interpretability
- Span distribution histogram
- Layer-wise span evolution

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
python run_all_4_adaptive_span.py \
    --models distilbert,roberta,gpt2 \
    --max_span 128 \
    --ramp_size 32 \
    --recovery_steps 50
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Max span $S_\text{max}$ | $\{32, 64, 128, 256\}$ |
| Ramp size $R$ | $\{1, 8, 16, 32, 64\}$ |
| L1 penalty $\lambda$ | $\{0, 0.001, 0.01, 0.1\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |

---