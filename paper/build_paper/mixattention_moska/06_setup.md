<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | FASA path |
|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Bidirectional |
| RoBERTa | 12 | 768 | 12 | 124.6M | Bidirectional |
| GPT-2 | 12 | 768 | 12 | 124.4M | Causal (no prune) |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size | Majority % |
|---|---|---|
| Train | 67,349 | 50.2% |
| Validation | 872 | 50.92% |

Sequence length truncated to 128.

## 6.3 FASA Configuration

| Parameter | Value |
|---|---|
| `fasa_keep_ratio` | 0.5 (default 0.5) |
| `importance_method` | 0.85 (`var`) |
| `protect_first_token` | 0.85 (True) |
| Min keep | 8 tokens |

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

## 6.5 Evaluation

- Accuracy, F1, Precision, Recall
- Total params
- Peak VRAM
- Inference time
- **Effective sequence length** (kept tokens per layer)

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
| VRAM | 8 GB GDDR6 |
| CUDA | 12.1 |
| PyTorch | 2.1.2 |

## 6.8 Reproduction

```bash
python run_all_4_fasa.py \
    --models distilbert,roberta,gpt2 \
    --fasa_keep_ratio 0.5 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 6.9 Seeds

Single seed (42).

## 6.10 Ablations

- Keep ratio: 0.25, 0.5, 0.75, 1.0
- Importance: `var` vs `l2`
- Protect first: on/off

## 6.11 Baselines

- `full` — unmodified model
- `fasa` — our method
- Prior modules: `cosformer`, `nha`, `a3`

---