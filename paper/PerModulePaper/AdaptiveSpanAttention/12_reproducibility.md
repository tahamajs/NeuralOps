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
python run_all_4_adaptive_span.py \
    --models distilbert,roberta,gpt2 \
    --max_span {{MAX_SPAN}} \
    --ramp_size {{RAMP_SIZE}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_adaptive_span.py --max_span {{MAX_SPAN}}
```

### Max Span Ablation (Table 7.5)
```bash
for S in 32 64 128 256; do
    python run_all_4_adaptive_span.py --max_span $S --output_suffix S$S
done
```

### Ramp Size Ablation (Table 7.6)
```bash
for R in 1 8 16 32 64; do
    python run_all_4_adaptive_span.py --ramp_size $R --output_suffix R$R
done
```

### L1 Penalty Ablation (Table 7.7)
```bash
for L in 0 0.001 0.01 0.1; do
    python run_all_4_adaptive_span.py --lambda_span $L --output_suffix L$L
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

| Model | Acc | Mean span |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_SPAN}} |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_SPAN}} |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_SPAN}} |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-adaptive_span`
- **Metrics:** `results/unified_adaptive_span/*/summary.csv`

## 12.7 Known Issues

- Slow training due to ramp allocation
- Fast path optimization needed for short sequences
- No hard truncation for actual speedup
```

---