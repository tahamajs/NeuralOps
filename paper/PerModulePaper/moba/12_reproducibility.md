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
python run_all_4_mamba.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,mamba \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_mamba.py
```

### Table 7.4 (Preservation gate ablation)
```bash
python run_all_4_mamba.py --pres_gate_init 0.0 --output_suffix g0
python run_all_4_mamba.py --pres_gate_init 2.0 --output_suffix g2
python run_all_4_mamba.py --pres_gate_init 4.0 --output_suffix g4
```

### Table 7.8 (Scaling)
```bash
for L in 64 128 256 512; do
    python run_all_4_mamba.py --max_seq_len $L --output_suffix L$L
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 73.62% ± 1% |
| RoBERTa | 84.17% ± 1% |
| GPT-2 | 50.92% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-mamba`
- **Metrics:** `results/unified_mamba/*/summary.csv`

---