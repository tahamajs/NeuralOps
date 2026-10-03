<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_primal.py \
    --models distilbert,roberta,gpt2 \
    --rank 64 \
    --pinv_iters 4 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_primal.py --rank 64
```

### Landmark Ablation (Table 7.4)
```bash
for m in 16 32 64 128 256; do
    python run_all_4_primal.py --rank $m --output_suffix m$m
done
```

### Pinv Iterations (Table 7.5)
```bash
for k in 0 2 4 6 8; do
    python run_all_4_primal.py --pinv_iters $k --output_suffix k$k
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

| Model | Acc | Note |
|---|---|---|
| DistilBERT | 90.14% ± 1% | — |
| RoBERTa | 68.00% ± 1% | — |
| GPT-2 | 63.07% ± 1% | — |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-primal`
- **Metrics:** `results/unified_primal/*/summary.csv`

## 12.7 Known Issues

- FP32 required for stable pseudoinverse
- Long sequences may OOM on 8 GB GPU
- Causal approximation not strict

---
