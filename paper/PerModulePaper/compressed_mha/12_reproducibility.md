# 12. Reproducibility

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |

## 12.3 Running Compressed MHA

```bash
python run_all_4_compressed_mha.py \
    --models distilbert,roberta,gpt2 \
    --rank_qkv {{RANK_QKV}} \
    --rank_out {{RANK_OUT}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,compressed_mha \
    --rank_qkv {{RANK_QKV}} \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_compressed_mha.py
```

### Table 7.5 (Rank Ablation)
```bash
for r in 32 64 128 256 384; do
    python run_all_4_compressed_mha.py --rank_qkv $r --rank_out $r --output_suffix r$r
done
```

### Table 7.6 (Joint vs Independent)
```bash
python run_all_4_compressed_mha.py --joint_svd false --output_suffix indep
```

### Table 7.9 (Nested Detection)
```bash
# Use the buggy version for comparison (not recommended)
python run_all_4_compressed_mha.py --legacy_nested_detection --output_suffix wrong
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_RATIO}}× |

## 12.8 Data Availability

- **Dataset:** SST-2 via `datasets`
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-compressed_mha`
- **Metrics:** `results/unified_compressed_mha/*/summary.csv`

---