# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

SST-2: 67,349 train / 872 val / 1,821 test. Majority baseline: 50.92%.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `use_fp8` | {{USE_FP8}} |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| `dropout` | 0.1 |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Hardware

| Component | Specification |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB GDDR6 |
| CUDA | {{CUDA_VERSION}} |
| PyTorch | {{TORCH_VERSION}} |
| FlashAttention | {{FA_STATUS}} |

**Critical.** On RTX 4070 (Ada Lovelace), FlashAttention-3 is not available 
(requires Hopper). Our wrapper uses SDPA fallback for all experiments.

## 6.6 Metrics

- Accuracy, F1
- Peak VRAM
- Inference time
- Parameter count

## 6.7 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.8 Reproduction

```bash
python run_all_4_flash3.py \
    --models distilbert,roberta,gpt2 \
    --use_fp8 {{USE_FP8}} \
    --lora_rank {{LORA_RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Unmodified model |
| `flash3` | Our wrapper |
| `flash3_fp8` | Wrapper + FP8 (Hopper only) |
| `sdpa_only` | Direct SDPA (no wrapper) |
