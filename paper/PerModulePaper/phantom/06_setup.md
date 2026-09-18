<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params | Attention |
|---|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M | Flat |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M | Nested |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M | Causal |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 2 | 800M | GQA + RoPE |

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| CUR rank $r$ | 32 |
| LSH rounds $R_\ell$ | 4 |
| LSH bucket size $k$ | 4 |
| Softpick enabled | True |
| Softpick temperature $\tau_s$ | 1.0 |
| Dropout | 0.1 |
| `max_error` $\tau$ | 0.024 |
| `fallback_to_original` | True |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy
- F1 (binary, macro)
- Precision, Recall
- Validation loss

### Efficiency
- Total parameters (before / after)
- Compression ratio
- Peak inference VRAM
- Inference time (median of 20 runs)
- **Average bucket size** (from LSH statistics)

### Compression
- Compressed layers count
- Fallback layers count
- Average reconstruction error
- Average bucket size

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
python run_all_4_phantom.py \
    --models distilbert,roberta,gpt2 \
    --cur_rank 32 \
    --lsh_rounds 4 \
    --recovery_steps 50 \
    --lora_rank 8
```

Runtime: ~PyTorch 2.0 min.

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed |
| `phantom` | Our method |
| `phantom_no_lsh` | Dense attention + CUR projections |
| `phantom_no_cur` | LSH attention + dense projections |
| `phantom_no_softpick` | Softmax instead of softpick |

## 6.10 Seeds

All experiments use seed 42. No multi-seed due to compute budget.

---