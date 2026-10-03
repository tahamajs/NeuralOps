# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running LoRA Attention

```bash
python run_all_4_lora_attention.py \
    --models distilbert,roberta,gpt2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --recovery_steps 50
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,lora_attention \
    --lora_rank 8 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_lora_attention.py
```

### Table 7.4 (Which projections)
```bash
# Q only
python run_all_4_lora_attention.py --targets q

# Q,V
python run_all_4_lora_attention.py --targets q,v

# All
python run_all_4_lora_attention.py --targets q,k,v,o
```

### Table 7.5 (Rank ablation)
```bash
for r in 4 8 16 32 64; do
    python run_all_4_lora_attention.py --lora_rank $r --output_suffix r$r
done
```

## 12.5 Merge Test

```python
from framework.attentions.lora_attention import merge_lora

# Before merge
out_before = model(x)

# Merge
merge_lora(model)

# After merge
out_after = model(x)
diff = (out_before - out_after).abs().max()
assert diff < 1e-4
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | Trainable |
|---|---|---|
| DistilBERT | 89.91% ± 1% | 0.85 |
| RoBERTa | 58.03% ± 1% | 0.85 |
| GPT-2 | 51.26% ± 1% | 0.85 |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-lora_attention`
- **Metrics:** `results/unified_lora_attention/*/summary.csv`

---