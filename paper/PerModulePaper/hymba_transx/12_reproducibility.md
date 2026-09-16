# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_hymba_transx.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,hymba_transx \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
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
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% | {{D_ALPHA}} |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% | {{R_ALPHA}} |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% | {{G_ALPHA}} |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-hymba_transx`.
- **Metrics:** `results/unified_hymba_transx/*/summary.csv`.
```

---