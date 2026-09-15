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

Base weights loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |
| Test | 1,821 | — | — |

Majority baseline: 50.92%. Sequence length: 128 tokens.

## 6.3 Calibration

We use the **default bundled corpus** (5 documents, ~215 tokens). This 
demonstrates AASVD's out-of-the-box usability. We additionally report 
results with task-specific calibration in Appendix C.

Calibration:
- Tokenizer: model's own
- Max length: 512
- Batch size: 1 (single forward pass)

## 6.4 Compression Hyperparameters

| Hyperparameter | Value |
|---|---|
| Rank $r$ | {{RANK}} |
| EMA $\beta$ | 0.9 |
| `max_error` $\tau$ | {{MAX_ERROR}} |
| `fallback_to_original` | True |
| `target_attention_only` | True |
| `use_fused_kernel` | False |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| LoRA rank | {{LORA_RANK}} |
| LoRA alpha | {{LORA_ALPHA}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.6 Evaluation

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

### Compression

- Compressed layers (count)
- Fallback layers (count)
- Average reconstruction error
- Per-layer error distribution

## 6.7 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

56% = majority + 3·SE.

## 6.8 Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| Transformers | {{HF_VERSION}} |

## 6.9 Reproduction

```bash
python run_all_4_aasvd.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} min.

## 6.10 Baselines

| Baseline | Description |
|---|---|
| `full` | Uncompressed model |
| `aasvd` | Our method |
| `aasvd_no_cov` | AASVD without covariance weighting |
| `aasvd_diag` | AASVD with diagonal-only weighting |
| Companion methods | `a3`, `sdrm`, `svd_linear` |
