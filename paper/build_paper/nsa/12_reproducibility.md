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
python run_all_4_nsa.py \
    --models distilbert,roberta,gpt2 \
    --window_size 128 \
    --top_k 2 \
    --recovery_steps 50
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
| DistilBERT | 90.83% ± 1% | ~50\%% |
| RoBERTa | 50.92% ± 1% | ~0.85% |
| GPT-2 | 50.92% ± 1% | ~0.85% |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-nsa`
- **Metrics:** `results/unified_nsa/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for topk stability
- No sparse kernels for actual speedup
- Very long sequences may OOM in topk

---