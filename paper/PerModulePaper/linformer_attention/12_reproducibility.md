# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Running Linformer

```bash
python run_all_4_linformer.py \
    --models distilbert,roberta,gpt2 \
    --linformer_k {{K}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,linformer \
    --linformer_k {{K}} \
    --recovery_steps {{RECOVERY_STEPS}}
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
| DistilBERT | {{D_ACC}}% ± 1% |
| RoBERTa | {{R_ACC}}% ± 1% |
| GPT-2 | {{G_ACC}}% ± 1% |

## 12.7 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-linformer`
- **Metrics:** `results/unified_linformer/*/summary.csv`

---
