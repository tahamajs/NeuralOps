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
python run_all_4_nsa.py \
    --models distilbert,roberta,gpt2 \
    --window_size {{WINDOW_SIZE}} \
    --top_k {{TOP_K}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_nsa.py --window_size 32 --top_k 16
```

### Window Ablation (Table 7.4)
```bash
for w in 8 16 32 64; do
    python run_all_4_nsa.py --window_size $w --output_suffix w$w
done
```

### Top-K Ablation (Table 7.5)
```bash
for k in 4 8 16 32; do
    python run_all_4_nsa.py --top_k $k --output_suffix k$k
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

| Model | Acc | Density |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_DENSITY}}% |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_DENSITY}}% |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_DENSITY}}% |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-nsa`
- **Metrics:** `results/unified_nsa/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for topk stability
- No sparse kernels for actual speedup
- Very long sequences may OOM in topk
```

---