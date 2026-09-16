# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/{{ORG}}/SVD_Project
cd SVD_Project
pip install -r requirements.txt
```

## 12.2 Running REAP-MoE

```bash
python run_all_4_reap_moe.py \
    --models distilbert,roberta,gpt2 \
    --n_experts {{N_EXPERTS}} \
    --top_k {{TOP_K}} \
    --recovery_steps {{RECOVERY_STEPS}}
```

Runtime: ~{{RUNTIME}} min.

## 12.3 Individual Model

```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,reap_moe \
    --n_experts {{N_EXPERTS}} \
    --top_k {{TOP_K}}
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
| DistilBERT | {{D_ACC}}% ± 1% | {{D_PARAM_CHANGE}} |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_PARAM_CHANGE}} |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_PARAM_CHANGE}} |

## 12.7 Data Availability

- **Dataset:** SST-2
- **Checkpoints:** `{{HF_ORG}}/svd-linear-attention-*-reap_moe`
- **Metrics:** `results/unified_reap_moe/*/summary.csv`

## 12.8 Known Issues

- Batch size 4 required for 12-layer models
- `PYTHONIOENCODING=utf-8` on Windows
```

---