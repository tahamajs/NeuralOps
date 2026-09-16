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
python run_all_4_chela_slay.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim {{FEATURE_DIM}} \
    --gate_init_bias {{GATE_INIT_BIAS}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --model_family bert \
    --modules full,chela_slay \
    --feature_dim {{FEATURE_DIM}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_chela_slay.py
```

### Table 7.5 (Feature Dim)
```bash
for f in 32 64 128 256; do
    python run_all_4_chela_slay.py --feature_dim $f --output_suffix f$f
done
```

### Table 7.6 (Gate Bias)
```bash
for bg in 0.0 2.0 5.0 10.0; do
    python run_all_4_chela_slay.py --gate_init_bias $bg --output_suffix gb$bg
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

| Model | Acc | Params Δ | Time Ratio |
|---|---|---|---|
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% | {{D_TIME_RATIO}}× |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% | {{R_TIME_RATIO}}× |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% | {{G_TIME_RATIO}}× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-chela_slay`.
- **Metrics:** `results/unified_chela_slay/*/summary.csv`.
```

---