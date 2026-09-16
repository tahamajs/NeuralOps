<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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
python run_all_4_nha.py \
    --models distilbert,roberta,gpt2 \
    --window_size {{WINDOW_SIZE}} \
    --rff_features {{RFF_FEATURES}} \
    --gate_init_bias {{GATE_BIAS}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_nha.py --window_size 64 --rff_features 128
```

### Window Size Ablation (Table 7.5)
```bash
for w in 16 32 64 128; do
    python run_all_4_nha.py --window_size $w --output_suffix w$w
done
```

### RFF Features Ablation (Table 7.6)
```bash
for m in 32 64 128 256; do
    python run_all_4_nha.py --rff_features $m --output_suffix m$m
done
```

### Gate Bias Ablation (Table 7.7)
```bash
for b in 0 1.0 1.8 3.0 5.0; do
    python run_all_4_nha.py --gate_init_bias $b --output_suffix b$b
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

| Model | Acc | Mean gate |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_GATE}} |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_GATE}} |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_GATE}} |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-nha`
- **Metrics:** `results/unified_nha/*/summary.csv`

## 12.7 Known Issues

- FP32 required for RFF stability
- Long sequences may need reduced chunk size
- RoPE fallback may fail on some models
```

---