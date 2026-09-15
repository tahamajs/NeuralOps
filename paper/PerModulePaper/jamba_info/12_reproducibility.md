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
python run_all_4_jamba_info.py \
    --models distilbert,roberta,gpt2 \
    --bottleneck_dim {{BOTTLENECK_DIM}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_jamba_info.py
```

### Table 7.3 (Bottleneck Dim)
```bash
for b in 64 128 256 512; do
    python run_all_4_jamba_info.py --bottleneck_dim $b --output_suffix b$b
done
```

### Table 7.4 (No Bottleneck)
```bash
python run_all_4_jamba_info.py --bottleneck_dim 768  # full dim
```

## 12.4 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.5 Expected Results

| Model | Acc | Params Δ |
|---|---|---|
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% |

## 12.6 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-jamba_info`.
- **Metrics:** `results/unified_jamba_info/*/summary.csv`.

---
