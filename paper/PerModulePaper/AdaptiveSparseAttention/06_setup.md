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

## 6.3 ASA Configuration

| Parameter | Value |
|---|---|
| Default sparsity (encoders) | {{SPARSITY_ENC}} |
| Default sparsity (decoders) | {{SPARSITY}} |
| Temperature $T$ | {{TEMPERATURE}} |
| Chunk size $C$ | {{CHUNK_SIZE}} |
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
- Inference time (median of 20 runs)
- Effective sparsity per head

### Sparsity Analysis
- Learned $\rho_h$ distribution
- Head budget histogram
- Attention mass retention

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
python run_all_4_adaptive_sparse.py \
    --models distilbert,roberta,gpt2 \
    --sparsity_ratio {{SPARSITY}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| Sparsity ratio | $\{0.1, 0.3, 0.5, 0.7, 0.9\}$ |
| Temperature $T$ | $\{0.01, 0.1, 0.5, 1.0\}$ |
| Chunk size $C$ | $\{32, 64, 128\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |
```

---