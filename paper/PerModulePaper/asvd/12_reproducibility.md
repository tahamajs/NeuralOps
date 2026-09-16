<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
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

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |

## 12.3 Running ASVD

```bash
python run_all_4_asvd.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --energy_threshold {{ENERGY_THRESHOLD}} \
    --error_threshold {{ERROR_THRESHOLD}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,asvd \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_asvd.py --models distilbert,roberta,gpt2
```

### Table 7.6 (Rank)
```bash
for r in 16 32 64 128; do
    python run_all_4_asvd.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (Energy threshold)
```bash
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_asvd.py --energy_threshold $e --output_suffix e$e
done
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% | {{G_RATIO}}× |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-asvd`
- Metrics: `results/unified_asvd/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- `raise` may abort injection on unusual sources
```

---