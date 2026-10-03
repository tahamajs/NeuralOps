<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params | Attention |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M | Causal |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

**Majority baseline:** 50.92%. **Sequence length:** 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `energy_threshold` $\tau$ | 0.92 | Cumulative energy cutoff |
| `svd_rank` (cap) | 288 | Maximum rank (rarely hit) |
| `residual_rank` $r_L$ | 16 | LoRA adapter rank |
| `min_rank` | 16 | Minimum rank floor |
| `trainable_svd` | False | Base frozen |
| `target_attention_only` | True | Only attention projections |

## 6.4 Calibration

**None required.** Adaptive SVD selects rank from the weight spectrum alone. 
No forward passes, no activation statistics.

## 6.5 Recovery Fine-Tuning

After compression, adapters are trained:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| LR schedule | Constant |
| Warmup | 0 steps |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |
| Trainable params | Only $A_P, B_P$ |

**Trainable tensor count:** 96 (2 per projection × 4 per 
layer × $N_l$).

## 6.6 Evaluation Metrics

### Quality
- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency
- Total parameters (before/after)
- Compression ratio
- Peak VRAM
- Inference time (median of 20 runs)
- Tokens/second

### Compression
- Compressed layer count
- Fallback layer count (if any)
- Per-projection rank
- Reconstruction error at rank $r_P$

## 6.7 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority + 3·SE. SE is $\sqrt{p(1-p)/N} \approx 1.69\%$ at $N = 872$.

## 6.8 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |
| Transformers | 4.44.0 |
| OS | Windows / Linux |

## 6.9 Reproduction

```bash
python run_all_4_adaptive_svd.py \
    --models distilbert,roberta,gpt2 \
    --energy_threshold 0.92 \
    --residual_rank 16 \
    --recovery_steps 50
```

**Expected runtime:** ~15 minutes on RTX 4070.

## 6.10 Seeds

All experiments use seed 42. No multi-seed runs due to compute budget.

## 6.11 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `adaptive_svd` | Our method |
| `svd_linear` | Fixed-rank SVD (rank 128) |
| `a3` | Fisher-weighted (from prior work) |
| `aasvd` | Full covariance (from prior work) |
