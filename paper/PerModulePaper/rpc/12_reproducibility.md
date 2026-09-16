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

## 12.3 Running RPC

```bash
python run_all_4_rpc.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,rpc \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_rpc.py --models distilbert,roberta,gpt2
```

### Table 7.4 (rank)
```bash
for r in 8 16 32 64; do
    python run_all_4_rpc.py --rank $r --output_suffix r$r
done
```

### Table 7.5 (LoRA rank)
```bash
for rl in 0 8 16 32; do
    python run_all_4_rpc.py --lora_rank $rl --output_suffix lora$rl
done
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

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | {{D_ACC}}% | +{{D_PARAM_PCT}}% |
| RoBERTa | {{R_ACC}}% | +{{R_PARAM_PCT}}% |
| GPT-2 | {{G_ACC}}% | +{{G_PARAM_PCT}}% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-rpc`
- Metrics: `results/unified_rpc/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- `raise` may abort injection on unusual sources
- Random projection seed affects results
```

---