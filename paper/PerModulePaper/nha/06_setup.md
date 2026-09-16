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

## 6.3 NHA Configuration

| Parameter | Value |
|---|---|
| Window size $w$ | {{WINDOW_SIZE}} |
| RFF features $m$ | {{RFF_FEATURES}} |
| Gate bias | {{GATE_BIAS}} |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Peak VRAM during forward
- Inference time (median of 20)
- Attention memory

### Gate Analysis
- Gate distribution histogram
- Per-position gate values
- Correlation with attention entropy

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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_nha.py \
    --models distilbert,roberta,gpt2 \
    --window_size {{WINDOW_SIZE}} \
    --rff_features {{RFF_FEATURES}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Window size $w$ | $\{16, 32, 64, 128\}$ |
| RFF features $m$ | $\{32, 64, 128, 256\}$ |
| Gate bias | $\{0, 1.0, 1.8, 3.0, 5.0\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |

---