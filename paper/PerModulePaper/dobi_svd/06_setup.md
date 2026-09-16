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
| Rank $r$ | {{RANK}} | Nominal rank |
| `init_rank_ratio` | {{INIT_RANK_RATIO}} | When `rank=None` |
| `init_gate_val` | {{INIT_GATE}} | Sigmoid init ≈ {{INIT_SIGMA}} |
| `max_error` | {{MAX_ERROR}} | Frobenius threshold |
| `max_retries` | 2 | Rank escalation |
| `fallback_to_original` | True | Safe fallback |
| `target_attention_only` | True | Only attention |
| Rank cap | 128 | Maximum rank |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
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
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.8 Reproduction

```bash
python run_all_4_dobi.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --init_gate_val {{INIT_GATE}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} minutes.

## 6.9 Seeds

All experiments use seed {{SEED}}.

## 6.10 Baselines

1. **`full`:** Uncompressed model.
2. **`dobi`:** Our method with learning gates.
3. **`dobi_pruned`:** Dobi with post-hoc gate pruning at τ_g=0.5.
4. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `mla`.
```

---