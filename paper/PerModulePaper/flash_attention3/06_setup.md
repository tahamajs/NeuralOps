# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params |
|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Pos | Neg |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max length: 128.

## 6.3 Environment

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop (Ampere) |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| FlashAttention | 1.0.0 |
| Transformers | 4.36.0 |

**Note.** Our test GPU is **Ampere (RTX 4070)**, not Hopper. FlashAttention-3 
is Hopper-only, so at test time our code likely falls back to **SDPA**. 
The reported results reflect **SDPA fallback**, which is functionally 
equivalent.

For Hopper testing, we provide separate scripts.

## 6.4 Hyperparameters

| Hyperparameter | Value |
|---|---|
| `lora_rank` | 8 |
| `lora_alpha` | 16 |
| `use_fp8` | True |

## 6.5 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 1e-4 |
| WD | 0.01 |
| Steps | 50 |
| Batch | 32 |
| Precision | FP16 |

## 6.6 Evaluation Metrics

### Quality
- Accuracy, F1, precision, recall, loss.

### Efficiency
- **Path used** (FA3 vs SDPA).
- Peak VRAM.
- Inference time.

### Attention Correctness
- Output difference between FA3 and SDPA (should be small).

## 6.7 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.8 Reproduction

```bash
python run_all_4_flash3.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `flash3` | Our method |
| `flash3_fp8` | FA3 with FP8 (Hopper only) |

## 6.10 Seeds

Single seed (42).

---