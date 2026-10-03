# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running LEKAv3

```bash
python run_all_4_leka_v3.py \
    --models distilbert,roberta,gpt2 \
    --num_landmarks 0.0 \
    --local_window_size 128 \
    --feature_dim 64 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,leka_v3 \
    --num_landmarks 0.0 \
    --local_window_size 128 \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_leka_v3.py
```

### Table 7.4 ($L_0$ Ablation)
```bash
for l0 in 8 16 32 64 128; do
    python run_all_4_leka_v3.py --num_landmarks $l0 --output_suffix l$l0
done
```

### Table 7.5 ($W$ Ablation)
```bash
for w in 8 16 32 64; do
    python run_all_4_leka_v3.py --local_window_size $w --output_suffix w$w
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

| Model | Acc |
|---|---|
| DistilBERT | 70.53% ± 1% |
| RoBERTa | 50.92% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-leka_v3`
- **Metrics:** `results/unified_leka_v3/*/summary.csv`

---