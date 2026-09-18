<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page (500 – 700 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |

## 12.3 Running Adaptive SVD

### Full Benchmark

```bash
python run_all_4_adaptive_svd.py \
    --models distilbert,roberta,gpt2 \
    --energy_threshold 0.92 \
    --residual_rank 16 \
    --recovery_steps 50
```

**Expected runtime:** ~15 minutes on RTX 4070.

### Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --dataset sst2 \
    --output_dir results/unified_adaptive_svd/distilbert \
    --parent_checkpoint results/unified_asvd/distilbert/finetuned_model.pt \
    --modules full,adaptive_svd \
    --rank 288 \
    --residual_rank 16 \
    --energy_threshold 0.92 \
    --recovery_steps 50
```

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"Hooshaai/svd-linear-attention-{model}-adaptive_svd",
        filename="weights.pt",
    )
    print(f"Downloaded {model} to {path}")
```

## 12.5 Reproducing Individual Tables

### Table 7.1 (Main)

```bash
python run_all_4_adaptive_svd.py - models distilbert,roberta,gpt2
# → results/unified_adaptive_svd/{model}/summary.csv
```

### Table 7.6 (Energy Threshold)

```bash
for tau in 0.85 0.90 0.92 0.95 0.99; do
    python run_all_4_adaptive_svd.py - energy_threshold $tau - output_suffix tau_$tau
done
```

### Table 7.7 (Residual Rank)

```bash
for rl in 0 4 8 16 32 64; do
    python run_all_4_adaptive_svd.py - residual_rank $rl - output_suffix rl_$rl
done
```

### Table 7.8 (Recovery Steps)

```bash
for steps in 10 25 50 100 200; do
    python run_all_4_adaptive_svd.py - recovery_steps $steps - output_suffix steps_$steps
done
```

## 12.6 Random Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 12.7 Expected Results

| Model | Acc | Ratio | Compress time |
|---|---|---|---|
| DistilBERT | 88.0% ± 1% | 3.0× | 15 s |
| RoBERTa | 74.0% ± 1% | 2.5× | 30 s |
| GPT-2 | 75.5% ± 1% | 2.5× | 30 s |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-adaptive_svd`
- **Metrics:** `results/unified_adaptive_svd/*/summary.csv`
- **Per-layer ranks:** in `metrics.json` under `layer_ranks`

## 12.9 Randomness and Determinism

- **SVD:** Deterministic
- **Rank selection:** Deterministic (given τ)
- **Adapter init:** Seeded random (default 42)
- **Recovery training:** Deterministic given seed

## 12.10 Known Issues

- **Windows encoding:** Set `PYTHONIOENCODING=utf-8`
- **CUDA version:** ≥ 11.8 for FP16 SVD
- **Checkpoint paths:** Some baselines auto-fallback
- **T5 verification:** Partial; use with care

## 12.11 Verification Steps

Before submitting, verify:

```bash
python -c "
from framework.attentions.adaptive_svd_attn import AdaptiveSVDAttention
print('Import OK')
"
```

Run one model to verify:

```bash
python run_all_4_adaptive_svd.py - models distilbert
```

Expected: DistilBERT passes quality gate (≥56%).
