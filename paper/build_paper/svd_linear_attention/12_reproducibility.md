<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility guide
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

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

## 12.3 Running SVD-Linear

```bash
python run_all_4_svd_linear_attention.py \
    --models distilbert,roberta,gpt2 \
    --rank 32 \
    --mode standard \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_svd_linear_attention.py --models distilbert,roberta,gpt2
```

### Table 7.6 (rank)
```bash
for r in 8 16 32 64 128; do
    python run_all_4_svd_linear_attention.py --rank $r --output_suffix r$r
done
```

### Table 7.7 (decay)
```bash
python run_all_4_svd_linear_attention.py --use_decay_gating false --output_suffix nodecay
python run_all_4_svd_linear_attention.py --use_decay_gating true  --output_suffix decay
```

## 12.5 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
```

## 12.6 Expected Results

| Model | Acc | Ratio |
|---|---|---|
| DistilBERT | 80.73% | 50\%× |
| RoBERTa | 50.92% | 2.1\times× |
| GPT-2 | 77.5% | 2.1\times× |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-svd_linear`
- Metrics: `results/unified_svd_linear_attention/*/summary.csv`

## 12.8 Streaming Example

```python
from framework.attentions.svd_linear_attention import SVDLinearAttention

# Initialize attention
attn = SVDLinearAttention(config, old_attn=src)
attn.eval()

# Streaming inference
state = None
for token in sequence:
    out, state = attn.step_recurrent(token, state)
```

## 12.9 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- Chunk size fixed at 64
- Decay init is approximate

---