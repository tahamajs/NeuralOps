# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running

```bash
python run_all_4_adaptive_svd.py \
    --models distilbert,roberta,gpt2 \
    --energy_threshold 0.92 \
    --residual_rank 16 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,adaptive_svd \
    --energy_threshold 0.92 \
    --residual_rank 16 \
    --recovery_steps 50
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
| DistilBERT | 87.96% | 2.4× | 160 |
| RoBERTa | 77.64% | 2.4× | 160 |
| GPT-2 | 76.38% | 2.4× | 160 |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-adaptive_svd`.
- **Metrics:** `results/unified_adaptive_svd/*/summary.csv`.
- **No calibration data required.**

---