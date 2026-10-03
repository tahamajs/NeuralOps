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

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 AdaSplash Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| $\alpha$ (entmax) | 1.5 | Sparsity parameter |
| $W$ (splash radius) | 32 | Window size |
| $S$ (sink count) | 4 | Fixed |
| $n_\text{iter}$ (bisection) | 25 | Fixed |
| $r_L$ (LoRA rank) | 16 | Adapter rank |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| Batch size | 8 |
| Precision | FP16 |
| Trainable params | LoRA only |

**Trainable tensor count:** 2 tensors per projection × 4 projections × $N_l$ 
layers = 8$N_l$ tensors.

## 6.5 Evaluation Metrics

### Quality
- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency
- Total parameters (before/after)
- Peak VRAM
- Inference time (median of 20 runs)
- Attention sparsity (% zero weights)

### Interpretability
- Attention entropy
- Effective support size
- Sink token mass

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |
| Transformers | 4.44.0 |

## 6.8 Reproduction

```bash
python run_all_4_adasplash.py \
    --models distilbert,roberta,gpt2 \
    --alpha 1.5 \
    --splash_radius 32 \
    --lora_rank 16 \
    --recovery_steps 50
```

Runtime: ~20 minutes.

## 6.9 Seeds

All experiments with seed 42.

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `adasplash` | Our method |
| `adasplash_a1` | α=1.0 (softmax) ablation |
| `adasplash_a2` | α=2.0 (sparsemax) ablation |
| `adasplash_no_splash` | No splash masking |
| `adasplash_no_sink` | No sink preservation |
