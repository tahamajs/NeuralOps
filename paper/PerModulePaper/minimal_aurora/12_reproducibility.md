# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | {{TORCH_VERSION}} |
| transformers | {{HF_VERSION}} |
| datasets | {{DATASETS_VERSION}} |

## 12.3 Running MinimalAURORA

```bash
python run_all_4_minimal_aurora.py \
    --models distilbert,roberta,gpt2 \
    --energy {{ENERGY}} \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2
```

### Table 7.5 (energy)
```bash
for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_minimal_aurora.py --energy $e --output_suffix e$e
done
```

### Table 7.7 (LoRA rank)
```bash
for rl in 0 8 16 32; do
    python run_all_4_minimal_aurora.py --lora_rank $rl --output_suffix lora$rl
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

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | {{D_ACC}}% | {{D_RATIO}}× |
| RoBERTa | {{R_ACC}}% | {{R_RATIO}}× |
| GPT-2 | {{G_ACC}}% | {{G_RATIO}}× |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-minimal_aurora`
- Metrics: `results/unified_minimal_aurora/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- `pres_gate` unused (documented)
- `output_scale` drift
- Silent injection failure
```

---