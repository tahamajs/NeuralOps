<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
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

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |
| numpy | {{NUMPY_VERSION}} |

## 12.3 Running EAT

```bash
python run_all_4_eat.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}} \
    --batch_size {{BATCH_SIZE}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,eat \
    --recovery_steps {{RECOVERY_STEPS}} \
    --lora_rank {{LORA_RANK}}
```

## 12.5 Reproducing Tables

### Table 7.1

```bash
python run_all_4_eat.py --models distilbert,roberta,gpt2
```

### Table 7.4 (LoRA rank)

```bash
for r in 4 8 16 32 64; do
    python run_all_4_eat.py --lora_rank $r --output_suffix lora$r
done
```

### Table 7.6 (Pruning)

```bash
python run_all_4_eat.py --prune_ratio 0.5
python run_all_4_eat.py --prune_ratio 0.7
python run_all_4_eat.py --no_prune
```

## 12.6 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ |
|---|---|---|
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% |

## 12.8 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-eat`
- Metrics: `results/unified_eat/*/summary.csv`
- Logs: `benchmark_logs/eat_*.log`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- RoBERTa may fail on some configs (see §8.2)
- Verify weight transfer with `sanity_check`

---
