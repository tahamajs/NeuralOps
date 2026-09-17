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

## 12.3 Running FUSIONATTN

```bash
python run_all_4_fusion_attn.py \
    --models distilbert,roberta,gpt2 \
    --block_size 64 \
    --top_k_blocks 2 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main
python run_all_4_fusion_attn.py --models distilbert,roberta,gpt2

# Block size sweep
for B in 32 64 128; do
    python run_all_4_fusion_attn.py --block_size $B --output_suffix B$B
done

# Top-K sweep
for K in 2 4 8; do
    python run_all_4_fusion_attn.py --top_k_blocks $K --output_suffix K$K
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
| DistilBERT | 86.4% ± 1% | Short |
| RoBERTa | 78.2% ± 1% | Short |
| GPT-2 | 77.5% ± 1% | Short |

## 12.7 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-fusion_attn`
- Metrics: `results/unified_fusion_attn/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- `inject_fusion_attn` raises on copy failure (recommend changing to return None)

---