# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Path |
|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | Galerkin |
| RoBERTa | 12 | 768 | 12 | Galerkin |
| GPT-2 | 12 | 768 | 12 | Fallback |

## 6.2 Dataset

SST-2: 67,349 / 872 / 1,821. Sequence length: 128.

## 6.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `rank` | {{RANK}} |
| `basis_type` | DCT-II |
| `lora_rank` | {{LORA_RANK}} |
| `lora_alpha` | {{LORA_ALPHA}} |
| Regularization $\epsilon$ | $10^{-4}$ |

## 6.4 Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Quality Gate

PASS ≥ 56%, PASS_LOW 50.92%–56%, FAIL < 50.92%.

## 6.6 Hardware

RTX 4070 Laptop, CUDA {{CUDA_VERSION}}, PyTorch {{TORCH_VERSION}}.

## 6.7 Reproduction

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 6.8 Baselines

| Baseline | Description |
|---|---|
| `full` | Unmodified model |
| `galerkin` | Our method |
| `galerkin_no_basis` | Disable projection (equivalent to LoRA) |
| `galerkin_rank_32` | Lower rank |
| `galerkin_rank_128` | Higher rank |
