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
python run_all_4_adaptive_sparse.py \
    --models distilbert,roberta,gpt2 \
    --sparsity_ratio 0.5 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_adaptive_sparse.py --sparsity_ratio 0.5
```

### Sparsity Ablation (Table 7.4)
```bash
for rho in 0.1 0.3 0.5 0.7 0.9; do
    python run_all_4_adaptive_sparse.py --sparsity_ratio $rho --output_suffix rho$rho
done
```

### Temperature Ablation (Table 7.5)
```bash
for T in 0.01 0.1 0.5 1.0; do
    python run_all_4_adaptive_sparse.py --temperature $T --output_suffix T$T
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

| Model | Acc | Sparsity |
|---|---|---|
| DistilBERT | 88.5% ± 1% | ~0.42 |
| RoBERTa | 76.0% ± 1% | ~0.45 |
| GPT-2 | 76.5% ± 1% | ~0.48 |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-adaptive_sparse`
- **Metrics:** `results/unified_adaptive_sparse/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for softmax stability
- Chunk boundary effects at small $C$
- RoPE fallback may fail on some models

---
