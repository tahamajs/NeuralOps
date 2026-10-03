<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 12.2 Running GQA

```bash
python run_all_4_gqa.py \
    --models distilbert,roberta,gpt2 \
    --num_kv_heads 6 \
    --recovery_steps 50 \
    --lora_rank 16
```

Runtime: ~15 minutes.

## 12.3 Reproducing Tables

### Main (Table 7.1)

```bash
python run_all_4_gqa.py --num_kv_heads 6
```

### KV Heads Ablation (Table 7.5)

```bash
for kv in 1 2 3 4 6 12; do
    python run_all_4_gqa.py --num_kv_heads $kv --output_suffix kv$kv
done
```

### LoRA Rank Ablation (Table 7.6)

```bash
for r in 4 8 16 32; do
    python run_all_4_gqa.py --lora_rank $r --output_suffix lr$r
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

| Model | Acc | KV ratio |
|---|---|---|
| DistilBERT | 89.91% ± 1% | 2.0× |
| RoBERTa | 84.40% ± 1% | 2.0× |
| GPT-2 | 82.00% ± 1% | 2.0× |

## 12.6 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `Hooshaai/svd-linear-attention-*-gqa`
- **Metrics:** `results/unified_gqa/*/summary.csv`

## 12.7 Known Issues

- Head count must divide cleanly
- RoPE applies to $K$ before repetition
- Windows: use UTF-8 encoding
