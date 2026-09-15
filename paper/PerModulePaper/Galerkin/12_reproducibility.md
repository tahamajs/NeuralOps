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

## 12.3 Running Galerkin

```bash
python run_all_4_galerkin.py \
    --models distilbert,roberta,gpt2 \
    --rank {{RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

## 12.4 Reproducing Tables

```bash
# Main
python run_all_4_galerkin.py --models distilbert,roberta,gpt2

# Rank ablation
for r in 16 32 64 128; do
    python run_all_4_galerkin.py --rank $r --output_suffix r$r
done
```

## 12.5 Seeds

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Path |
|---|---|---|
| DistilBERT | {{D_ACC}}% ± 1% | Galerkin |
| RoBERTa | {{R_ACC}}% ± 1% | Galerkin |
| GPT-2 | {{G_ACC}}% ± 1% | Fallback |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-galerkin`
- Metrics: `results/unified_galerkin/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- GPT-2 uses fallback (no Galerkin benefit)
