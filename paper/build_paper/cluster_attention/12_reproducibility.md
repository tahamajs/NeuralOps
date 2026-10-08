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

## 12.3 Running ClusterFormer

```bash
python run_all_4_cluster.py \
    --models distilbert,roberta,gpt2 \
    --num_clusters 16 \
    --window_size 32 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main results
python run_all_4_cluster.py --models distilbert,roberta,gpt2

# K ablation
for K in 4 8 16 32 64; do
    python run_all_4_cluster.py --num_clusters $K --output_suffix K$K
done

# Window ablation
for W in 0 8 32 64; do
    python run_all_4_cluster.py --window_size $W --output_suffix W$W
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

| Model | Acc | Params Δ |
|---|---|---|
| DistilBERT | 90.02% ± 1% | +589826M |
| RoBERTa | 72.02% ± 1% | +1179648M |
| GPT-2 | 87.96% ± 1% | +1188876M |

## 12.7 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-cluster`
- Metrics: `results/unified_cluster/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- K-means cdist may be slow for very long sequences

---