# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |

## 12.3 Running

```bash
python run_all_4_adaptive_svd.py \
    --models distilbert,roberta,gpt2 \
    --energy_threshold {{ENERGY_THRESHOLD}} \
    --residual_rank {{RESIDUAL_RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,adaptive_svd \
    --energy_threshold {{ENERGY_THRESHOLD}} \
    --residual_rank {{RESIDUAL_RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_adaptive_svd.py
```

### Table 7.4 (Energy τ)
```bash
for tau in 0.80 0.85 0.92 0.95 0.99; do
    python run_all_4_adaptive_svd.py --energy_threshold $tau --output_suffix tau$tau
done
```

### Table 7.5 (Residual Rank)
```bash
for r in 0 8 16 32; do
    python run_all_4_adaptive_svd.py --residual_rank $r --output_suffix ra$r
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | Attention Ratio | Avg Rank |
|---|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_RATIO}}× | {{AVG_RANK}} |
| RoBERTa | {{R_ACC}}% | {{R_RATIO}}× | {{R_AVG_RANK}} |
| GPT-2 | {{G_ACC}}% | {{G_RATIO}}× | {{G_AVG_RANK}} |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-adaptive_svd`.
- **Metrics:** `results/unified_adaptive_svd/*/summary.csv`.
- **No calibration data required.**

---
