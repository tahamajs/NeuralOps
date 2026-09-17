<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV Heads | Params |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |
| Test | 1,821 | — |

Sequence length truncated to 128.

## 6.3 FP8 Configuration

| Parameter | Value |
|---|---|
| `use_fp8` | True |
| FP8 format | e4m3fn |
| Fallback dtype | FP16 |
| `FLASH_ATTN_AVAILABLE` | **0.85** |
| FP8 path taken? | **0.85** |

## 6.4 Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Recovery steps | 50 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Batch size | 32 |
| Precision | FP16 |
| Dropout | 0.1 |

## 6.5 Evaluation

### 6.5.1 Quality Metrics

- Accuracy
- F1 (binary), F1 (macro)
- Precision, Recall

### 6.5.2 Efficiency Metrics

- Total params
- Trainable params
- Peak VRAM
- Inference time (median of 20 runs)
- Throughput (tokens/sec)

### 6.5.3 Hardware Detection Metrics

- `FLASH_ATTN_AVAILABLE` — whether flash-attn is importable
- `FP8_TAKEN` — whether the FP8 path was used at least once
- `SDPA_BACKEND` — which SDPA backend PyTorch dispatched to

## 6.6 Quality Gate

| Level | Threshold |
|---|---|
| PASS | ≥ 56.0% |
| PASS_LOW | 50.92% – 56.0% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop |
| GPU VRAM | 8 GB GDDR6 |
| GPU Architecture | Ada Lovelace (SM 8.9) |
| FP8 Tensor Cores | Available architecturally |
| FP8 Exposed? | **0.85** |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |
| Transformers | 4.36.0 |
| flash-attn | **1.0.0** |

## 6.8 Why FP8 May Not Be Available

On RTX 4070 Laptop:

1. FP8 tensor cores exist in silicon (Ada architecture).
2. But the consumer driver may not expose them to CUDA.
3. PyTorch's `float8_e4m3fn` dtype exists in the build.
4. But `flash_attn_func` may reject the call.

Our code catches these failures and falls back. **The result is that on 
our hardware, the FP8 path is not exercised**, and we report results from 
the SDPA fallback.

This is documented as a **key limitation** (Section 10).

## 6.9 Reproduction

```bash
python run_all_4_flash3_fp8.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

Expected runtime: ~PyTorch 2.0 minutes on RTX 4070.

## 6.10 Seeds

Single seed (42). No multi-seed runs.

## 6.11 Baselines

- `full` — unmodified model
- `flash3_fp8` — our implementation
- Prior modules: `nha`, `nasc`, `a3`, etc.

---