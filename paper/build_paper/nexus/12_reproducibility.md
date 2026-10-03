# 12. Reproducibility

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
python run_all_4_nexus.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,nexus \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_nexus.py
```

### Table 7.3 (Block size)
```bash
for bs in 8 16 32 64; do
    python run_all_4_nexus.py --block_size $bs --output_suffix bs$bs
done
```

### Table 7.4 (Top-k)
```bash
for k in 1 2 4 8; do
    python run_all_4_nexus.py --top_k_blocks $k --output_suffix k$k
done
```

## 12.5 Enabling Inactive Components

```bash
# Enable stripe (needs N > 256)
python run_all_4_nexus.py --max_seq_len 512 --stripe_enabled true

# Enable RFF rectification
python run_all_4_nexus.py --enable_spectraformer true
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 50.92% ± 1% |
| RoBERTa | 50.92% ± 1% |
| GPT-2 | 51.61% ± 1% |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-nexus`
- **Metrics:** `results/unified_nexus/*/summary.csv`

---