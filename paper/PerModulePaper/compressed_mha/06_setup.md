# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | KV heads | Params |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 12 | 66.9M |
| RoBERTa-base | 12 | 768 | 12 | 12 | 124.6M |
| GPT-2 | 12 | 768 | 12 | 12 | 124.4M |

## 6.2 Dataset

**SST-2**:

| Split | Size | Positives | Negatives |
|---|---|---|---|
| Train | 67,349 | 33,807 | 33,542 |
| Validation | 872 | 444 | 428 |

Majority baseline: 50.92%. Max sequence length: 128.

## 6.3 Compressed MHA Hyperparameters

| Hyperparameter | Default |
|---|---|
| `rank_qkv` | 64 |
| `rank_out` | 64 |
| `lora_rank` | 16 |
| `dropout` | 0.1 |

**Default behavior:** If no explicit rank, the code promotes `rank_qkv` to 
`min(384, all_head + 2*kv_head, hidden)`. For $d = 768$, this is 384.

## 6.4 LoRA Recovery

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-4 |
| Weight decay | 1e-2 |
| Steps | 50 |
| Batch size | 8 |
| Precision | FP16 |

## 6.5 Evaluation

### Quality
- Accuracy, F1, precision, recall, loss

### Efficiency
- Total parameters (before/after)
- Compression ratio
- Peak VRAM
- Inference time

### Compression
- Effective SVD ranks
- Reconstruction error per layer
- Fallback count

## 6.6 Quality Gate

| Status | Accuracy |
|---|---|
| PASS | ≥ 56% |
| PASS_LOW | 50.92–56% |
| FAIL | < 50.92% |

## 6.7 Hardware

| Component | Spec |
|---|---|
| GPU | RTX 4070 Laptop |
| VRAM | 8 GB |
| CUDA | 12.1 |
| PyTorch | 2.4.0 |

## 6.8 Reproduction

```bash
python run_all_4_compressed_mha.py \
    --models distilbert,roberta,gpt2 \
    --rank_qkv 64 \
    --rank_out 64 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 6.9 Baselines

| Baseline | Description |
|---|---|
| `full` | Standard attention |
| `compressed_mha` | Our method |
| `compressed_mha_independent` | Independent SVD (ablation) |
| `compressed_mha_no_lora` | No LoRA recovery (ablation) |

## 6.10 Seeds

Single seed (42). Limitation stated in Section 10.

---