# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running Linformer

```bash
python run_all_4_linformer.py \
    --models distilbert,roberta,gpt2 \
    --linformer_k 4 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,linformer \
    --linformer_k 4 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_linformer.py
```

### Table 7.4 (k ablation)
```bash
for k in 16 32 64 128 256; do
    python run_all_4_linformer.py --linformer_k $k --output_suffix k$k
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
| DistilBERT | 50.92% ± 1% |
| RoBERTa | 48.17% ± 1% |
| GPT-2 | 77.5% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-linformer`
- **Metrics:** `results/unified_linformer/*/summary.csv`

---