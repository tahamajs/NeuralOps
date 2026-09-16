<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 06_setup.md
     PURPOSE: Experimental setup
     DEPENDENCIES: 05_architecture.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 6. Experimental Setup

## 6.1 Models

| Model | Layers | Hidden | Heads | Params | Attention | Causal |
|---|---|---|---|---|---|---|
| DistilBERT | 6 | 768 | 12 | 66.9M | Flat | No |
| RoBERTa | 12 | 768 | 12 | 124.6M | Nested | No |
| GPT-2 | 12 | 768 | 12 | 124.4M | Conv1D | Yes |

## 6.2 Dataset

**SST-2** (Socher et al., 2013):

| Split | Size |
|---|---|
| Train | 67,349 |
| Validation | 872 |
| Test | 1,821 |

**Sequence length:** 128 (below Nyström activation threshold).

Majority baseline: 50.92%.

## 6.3 Nyströmformer Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `num_landmarks` | {{NUM_LANDMARKS}} | Default 32 |
| `pinv_iterations` | 3 | Newton-Schulz |
| Short seq threshold | 128 | Dense fallback |
| `lora_rank` | {{LORA_RANK}} | Recovery |
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
- **Path used** (dense or Nyström)

### Approximation Quality
- Nyström approximation error (when active)

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
python run_all_4_nystrom.py \
    --models distilbert,roberta,gpt2 \
    --num_landmarks {{NUM_LANDMARKS}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

Runtime: ~{{RUNTIME}} minutes.

## 6.9 Long Sequence Evaluation

**Note.** At $N = 128$, all models use dense fallback. To activate Nyström, 
sequences must be $N > 128$. We test this in a separate experiment with 
$N = 512$:

```bash
python run_all_4_nystrom.py \
    --models distilbert \
    --seq_len 512 \
    --num_landmarks 32
```

## 6.10 Seeds

All experiments use seed {{SEED}}.

## 6.11 Baselines

1. **`full`:** Uncompressed model.
2. **`nystrom`:** Our method (dense fallback at $N=128$).
3. **Companion methods:** `minimal_svd`, `svd_llm_v2`, `mla`, `dobi`.

---