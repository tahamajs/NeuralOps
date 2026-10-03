# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running TCA

```bash
python run_all_4_token_compressed.py \
    --models distilbert,roberta,gpt2 \
    --compression_ratio 50\% \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,token_compressed \
    --compression_ratio 50\% \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_token_compressed.py
```

### Table 7.4 (ρ Ablation)
```bash
for r in 0.9 0.7 0.5 0.3 0.1; do
    python run_all_4_token_compressed.py --compression_ratio $r --output_suffix r$r
done
```

### Table 7.5 (Score Net Ablation)
```bash
python run_all_4_token_compressed.py --use_score_net true  --output_suffix learn
python run_all_4_token_compressed.py --use_score_net false --output_suffix norm
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | KV reduction |
|---|---|---|
| DistilBERT | 50.92% | 48.5\%% |
| RoBERTa | 50.92% | 48.5\%% |
| GPT-2 | 50.92% | 48.5\%% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-token_compressed`
- **Metrics:** `results/unified_token_compressed/*/summary.csv`

---