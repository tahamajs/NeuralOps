<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.4.0 |
| transformers | 4.44.0 |
| datasets | 2.20.0 |

## 12.3 Running OHA

```bash
python run_all_4_oha.py \
    --models distilbert,roberta,gpt2 \
    --window_size 64 \
    --compression_rank 32 \
    --recovery_steps 50
```

## 12.4 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,oha \
    --recovery_steps 50
```

## 12.5 Reproducing Tables

### Table 7.1
```bash
python run_all_4_oha.py --models distilbert,roberta,gpt2
```

### Table 7.4 (compress rank)
```bash
for dc in 8 16 32 64; do
    python run_all_4_oha.py --compression_rank $dc --output_suffix dc$dc
done
```

### Table 7.5 (window)
```bash
for w in 16 32 64 128; do
    python run_all_4_oha.py --window_size $w --output_suffix w$w
done
```

### Table 7.6 (gate bias)
```bash
python run_all_4_oha.py --gate_bias 0.0
python run_all_4_oha.py --gate_bias 1.0
python run_all_4_oha.py --gate_bias 2.0
python run_all_4_oha.py --gate_bias 3.0
```

## 12.6 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.7 Expected Results

| Model | Acc | Param Δ % |
|---|---|---|
| DistilBERT | 89.2% | +1.40% |
| RoBERTa | 76.5% | +1.51% |
| GPT-2 | 77.8% | +1.51% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `Hooshaai/svd-linear-attention-*-oha`
- Metrics: `results/unified_oha/*/summary.csv`

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8` recommended
- Cross-attention not supported
- `raise` in copy_weights may abort injection

---
