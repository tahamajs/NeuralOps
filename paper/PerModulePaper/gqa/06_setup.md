<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | H | d_h | Params | Source KV |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 12 | 64 | 66.9M | 12 (MHA) |
| RoBERTa-base | 12 | 12 | 64 | 124.6M | 12 (MHA) |
| GPT-2 | 12 | 12 | 64 | 124.4M | 12 (MHA) |

All models loaded from `results/*_baseline/finetuned_model.pt`.

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Majority baseline: 50.92%. Sequence length: 128.

## 6.3 GQA Configuration

| Parameter | Value |
|---|---|
| Target $H_{KV}$ | {{KV_HEADS}} |
| Reduction factor | $H / G = {{REDUCTION}}$ |
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
- KV cache at inference (per token, per layer)
- Peak VRAM during forward pass
- Inference time (median of 20)

### KV Cache Calculation

```python
kv_bytes_per_token_per_layer = 2 * G * d_h * 2  # FP16
total_kv = N_layers * L * B * kv_bytes_per_token_per_layer
```

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
python run_all_4_gqa.py \
    --models distilbert,roberta,gpt2 \
    --num_kv_heads {{KV_HEADS}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 6.9 Ablations

| Ablation | Values |
|---|---|
| $G$ | $\{1, 2, 3, 4, 6, 12\}$ |
| LoRA rank | $\{4, 8, 16, 32\}$ |
| Recovery steps | $\{10, 25, 50, 100\}$ |
