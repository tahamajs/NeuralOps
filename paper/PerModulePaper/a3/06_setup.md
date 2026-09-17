<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages (600 - 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

We evaluate on three pretrained backbones:

| Model | Layers | Hidden | Params | Attention |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 124.4M | Causal |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 (50.2%) | 33,542 |
| Validation | 872 | 444 (50.9%) | 428 |
| Test | 1,821 | — | — |

Majority baseline: **50.92%**.

Sequence length truncated to 128 tokens.

## 6.3 Calibration Data

For A3's activation statistics, we use the **training set** as calibration 
source. This is legitimate because:

- The calibration pass is a forward-only pass (no gradient)
- It uses unlabeled inputs (labels not required)
- It is not a form of test set leakage

We use **1000 examples** (a subsample of the training set) for 
calibration to keep memory bounded.

## 6.4 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Rank $r$ | 64 | Target rank |
| `use_fisher` | True | Activation-aware weighting |
| `recalibrate_ln` | True | LayerNorm recalibration |
| `max_error` $\tau$ | 0.20 | Frobenius threshold |
| `max_retries` | 3 | Adaptive escalation steps |
| `fallback_to_original` | True | Safe fallback |
| `normalize_scale` | True | Clamp weights |
| `target_attention_only` | False | Compress all layers |

## 6.5 Recovery Fine-Tuning

After A3 compression, we train LoRA adapters:

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 1e-2 |
| Recovery steps | 50 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 8 |
| Precision | FP16 |

## 6.6 Evaluation Metrics

### 6.6.1 Quality Metrics

- **Accuracy:** $\text{Acc} = \frac{\text{correct}}{\text{total}}$
- **F1:** Binary and macro-averaged
- **Precision, Recall**

### 6.6.2 Efficiency Metrics

- **Total parameters:** `sum(p.numel() for p in model.parameters())`
- **Trainable parameters**
- **Peak VRAM:** `torch.cuda.max_memory_allocated()`
- **Inference time:** Median of 20 forward passes
- **Throughput:** Tokens/sec

### 6.6.3 Compression Metrics

- **Compression ratio:** Baseline params / compressed params
- **Reconstruction error:** $\|W - UV\|_F / \|W\|_F$
- **Compressed layers:** Count of successfully factorized layers
- **Fallback layers:** Count of layers preserved

## 6.7 Quality Gate

| Level | Accuracy threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% corresponds to majority baseline + 3×SE.

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
python run_all_4_a3.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --recovery_steps 50 \
    --lora_rank 16
```

Expected runtime: ~15 minutes.

## 6.10 Seeds

All experiments use seed 42. No multi-seed runs due to compute budget.

## 6.11 Baselines

We compare against:

1. **`full`:** Uncompressed pretrained model.
2. **`a3`:** Our method.
3. **`a3_no_fisher`:** A3 without Fisher weighting (ablations).
4. **`a3_no_ln`:** A3 without LayerNorm recalibration (ablations).
5. **Prior modules** (from our benchmark suite):
   - `sdrm`, `svd_linear`, `nha`, etc.
