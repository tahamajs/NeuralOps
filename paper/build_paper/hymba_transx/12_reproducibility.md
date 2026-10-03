# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_hymba_transx.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hymba_transx \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_hymba_transx.py
```

### Table 7.4 (Differential Ablation)
```bash
# Standard: modify module to use single Q-K
# Differential: default
```

### Table 7.5 (SSM Ablation)
```bash
python run_all_4_hymba_transx.py --alpha 1.0  # attn only
python run_all_4_hymba_transx.py --alpha 0.0  # ssm only
python run_all_4_hymba_transx.py --alpha 0.5  # fixed
# default: learned
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Params Δ | α |
|---|---|---|---|
| DistilBERT | 49.08% | +48.5\%% | 0.85 |
| RoBERTa | 51.15% | +48.5\%% | 0.85 |
| GPT-2 | 57.57% | +48.5\%% | 0.85 |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-hymba_transx`.
- **Metrics:** `results/unified_hymba_transx/*/summary.csv`.

---