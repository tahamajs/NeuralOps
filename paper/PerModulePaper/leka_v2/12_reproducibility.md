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
python run_all_4_leka_v2.py \
    --models distilbert,roberta,gpt2 \
    --feature_dim {{FEATURE_DIM}} \
    --landmark_ratio {{LANDMARK_RATIO}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_leka_v2.py
```

### Table 7.4 (ρ Ablation)
```bash
for rho in 0.25 0.5 0.75 1.0; do
    python run_all_4_leka_v2.py --landmark_ratio $rho --output_suffix rho$rho
done
```

### Table 7.5 (Feature dim)
```bash
for f in 64 128 256 512; do
    python run_all_4_leka_v2.py --feature_dim $f --output_suffix f$f
done
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

| Model | Acc | Landmark ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_LR}} |
| RoBERTa | {{R_ACC}}% | {{R_LR}} |
| GPT-2 | {{G_ACC}}% | {{G_LR}} |

## 12.6 Data Availability

- **Dataset:** SST-2.
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-leka_v2`.
- **Metrics:** `results/unified_leka_v2/*/summary.csv`.
```

---