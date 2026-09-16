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
python run_all_4_adaptive_sparse.py \
    --models distilbert,roberta,gpt2 \
    --sparsity_ratio {{SPARSITY}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
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
| DistilBERT | {{D_ACC}}% ± 1% | ~{{D_SP}} |
| RoBERTa | {{R_ACC}}% ± 1% | ~{{R_SP}} |
| GPT-2 | {{G_ACC}}% ± 1% | ~{{G_SP}} |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-adaptive_sparse`
- **Metrics:** `results/unified_adaptive_sparse/*/summary.csv`

## 12.7 Known Issues

- FP32 needed for softmax stability
- Chunk boundary effects at small $C$
- RoPE fallback may fail on some models

---