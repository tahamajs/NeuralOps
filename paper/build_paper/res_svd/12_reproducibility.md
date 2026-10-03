# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Dependencies

| Package | Version |
|---|---|
| torch | 2.1.2 |
| transformers | 4.36.0 |
| datasets | 1.0.0 |

## 12.3 Running Res-SVD

```bash
python run_all_4_res_svd.py \
    --models distilbert,roberta,gpt2 \
    --base_rank 32 \
    --residual_rank 16 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

### Table 7.1
```bash
python run_all_4_res_svd.py --models distilbert,roberta,gpt2
```

### Table 7.4 (base rank)
```bash
for rb in 8 16 32 64; do
    python run_all_4_res_svd.py --base_rank $rb --output_suffix rb$rb
done
```

### Table 7.5 (residual rank)
```bash
for rr in 0 2 4 8 16; do
    python run_all_4_res_svd.py --residual_rank $rr --output_suffix rr$rr
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
| DistilBERT | 89.56% | 50\%× |
| RoBERTa | 76.61% | 2.1\times× |
| GPT-2 | 71.33% | 2.1\times× |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-res_svd`
- Metrics: `results/unified_res_svd/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- Cross-attention not supported
- Silent injection fallback

---