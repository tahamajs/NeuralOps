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
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal Conv1D |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: **50.92%**.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Rank $r$ | 32 | Nominal rank |
| `init_rank_ratio` | 2.1\times | When `rank=None` |
| `init_gate_val` | 0.5 | Sigmoid init ≈ 1.0 |
| `max_error` | 0.024 | Frobenius threshold |
| `max_retries` | 2 | Rank escalation |
| `fallback_to_original` | True | Safe fallback |
| `target_attention_only` | True | Only attention |
| Rank cap | 128 | Maximum rank |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| Batch size | 32 |
| Precision | FP16 |
| Trainable | Gates + LoRA |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, Precision, Recall

### Compression
- Nominal rank $r$
- **Effective rank** $r_\text{eff} = \#\{i : \sigma(\alpha_i) > \tau_g\}$
- **Effective parameter ratio** (using $r_\text{eff}$)
- Reconstruction error

### Efficiency
- Peak inference VRAM
- Inference time
- Total parameters

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
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |

## 6.8 Reproduction

```bash
python run_all_4_dobi.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --init_gate_val 0.5 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 minutes.

## 6.9 Seeds

All experiments use seed 42.

## 6.10 Baselines

1. **`full`:** Uncompressed model.
2. **`dobi`:** Our method with learning gates.
3. **`dobi_pruned`:** Dobi with post-hoc gate pruning at τ_g=0.5.
4. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `mla`.

---