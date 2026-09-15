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
python run_all_4_ptp.py \
    --models distilbert,roberta,gpt2 \
    --keep_ratio {{KEEP_RATIO}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_ptp.py --keep_ratio 0.8
```

### Keep Ratio Ablation (Table 7.3)
```bash
for rho in 0.5 0.7 0.8 0.9 0.95; do
    python run_all_4_ptp.py --keep_ratio $rho --output_suffix rho$rho
done
```

### [CLS] Ablation (Table 7.4)
```bash
python run_all_4_ptp.py --preserve_cls true --output_suffix cls
python run_all_4_ptp.py --preserve_cls false --output_suffix nopcls
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

| Model | Acc | Kept |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_KEPT}}% |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_KEPT}}% |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_KEPT}}% |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-ptp`
- **Metrics:** `results/unified_ptp/*/summary.csv`

## 12.7 Known Issues

- Score network sees future tokens (not for generation)
- Very short sequences lose accuracy
- Shared selection across queries
