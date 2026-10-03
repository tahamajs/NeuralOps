<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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
python run_all_4_ptp.py \
    --models distilbert,roberta,gpt2 \
    --keep_ratio 0.5 \
    --recovery_steps 50
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
| DistilBERT | 86.4% ± 1% | ~0.85% |
| RoBERTa | 78.2% ± 1% | ~0.85% |
| GPT-2 | 77.5% ± 1% | ~0.85% |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-ptp`
- **Metrics:** `results/unified_ptp/*/summary.csv`

## 12.7 Known Issues

- Score network sees future tokens (not for generation)
- Very short sequences lose accuracy
- Shared selection across queries

---