# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Running REAP-MoE

```bash
python run_all_4_reap_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts 8 \
    --top_k 2 \
    --recovery_steps 50
```

Runtime: ~PyTorch 2.0 min.

## 12.3 Individual Model

```bash
python neuralops/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,reap_moe \
    --n_experts 8 \
    --top_k 2
```

## 12.4 Reproducing Tables

### Table 7.1 (Main)
```bash
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2
```

### Table 7.5 (Experts)
```bash
for E in 2 4 8 16; do
    python run_all_4_reap_moe.py --n_experts $E --output_suffix E$E
done
```

### Table 7.6 (Top-k)
```bash
for k in 1 2 4; do
    python run_all_4_reap_moe.py --top_k $k --output_suffix k$k
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
| DistilBERT | 90.02% ± 1% | 66.4M |
| RoBERTa | 50.92% ± 1% | 66.4M |
| GPT-2 | 52.41% ± 1% | 66.4M |

## 12.7 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `efficiency-lab/svd-linear-attention-*-reap_moe`
- **Metrics:** `results/unified_reap_moe/*/summary.csv`

## 12.8 Known Issues

- Batch size 4 required for 12-layer models
- `PYTHONIOENCODING=utf-8` on Windows

---