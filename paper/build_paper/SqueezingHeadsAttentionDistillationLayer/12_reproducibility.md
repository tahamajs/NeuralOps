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
python run_all_4_squeezing_heads.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank 16 \
    --head_dim_reduction 1.0 \
    --recovery_steps 50
```

## 12.3 Reproducing Tables

### Main (Table 7.1)
```bash
python run_all_4_squeezing_heads.py - lora_rank 16
```

### LoRA Rank Ablation (Table 7.4)
```bash
for r in 4 8 16 32; do
    python run_all_4_squeezing_heads.py - lora_rank $r - output_suffix r$r
done
```

### Head-Dim Reduction (Table 7.2)
```bash
for s in 1.0 1.5 2.0 2.5; do
    python run_all_4_squeezing_heads.py - head_dim_reduction $s - output_suffix s$s
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

| Model | Acc | Trainable |
|---|---|---|
| DistilBERT | 90.25% ± 1% | ~589824 |
| RoBERTa | 76.8% ± 1% | ~1179648 |
| GPT-2 | 77.5% ± 1% | ~1179648 |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-squeezing_heads`
- **Metrics:** `results/unified_squeezing_heads/*/summary.csv`

## 12.7 Known Issues

- FP32 attention scores for stability
- LoRA folding required for speedup
- Very small head-dim loses information irrecoverably

---
