<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | RoPE |
|---|---|---|---|---|---|
| DistilBERT-base | 6 | 768 | 12 | 66.9M | Added |
| RoBERTa-base | 12 | 768 | 12 | 124.6M | Added |
| GPT-2 | 12 | 768 | 12 | 124.4M | Added |
| Qwen3.5-0.8B | 24 | 1024 | 8 | 800M | Native |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

Sequence length: 128. Majority baseline: 50.92%.

## 6.3 UnifiedRoPE Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `max_seq_len` | 4096 | RoPE cache length |
| `base` | 10000 | RoPE frequency base |
| `rotary_dim` | `2 * (head_dim // 2)` | Full rotation |
| `lora_rank` | {{LORA_RANK}} | LoRA adapters |
| `lora_alpha` | {{LORA_ALPHA}} | Scaling |
| `dropout` | 0.1 | Standard |

## 6.4 Recovery Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | {{LR}} |
| Weight decay | {{WD}} |
| Recovery steps | {{RECOVERY_STEPS}} |
| Batch size | {{BATCH_SIZE}} |
| Precision | FP16 |

## 6.5 Evaluation Metrics

### Quality
- Accuracy, F1, Precision, Recall

### Efficiency
- Inference time
- Peak VRAM

### RoPE Correctness
- Norm preservation: $\|\tilde{q}\|_2 = \|q\|_2$
- Relative position: `q_i @ k_j` depends only on `i - j`

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
python run_all_4_unified_rope.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} minutes.

## 6.9 Seeds

All experiments use seed {{SEED}}.

## 6.10 Baselines

1. **`full`:** Uncompressed model (original attention).
2. **`unified_rope`:** Our module.
3. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `dobi`.

---