<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_nasc.py \
    --models distilbert,roberta,gpt2 \
    --window_size 128 \
    --block_size 64 \
    --top_k 2 \
    --rff_features 64 \
    --recovery_steps 50
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download
for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-nasc",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Main Results

```bash
python run_all_4_nasc.py --models distilbert,roberta,gpt2
```

### Encoder Cascade Ablation

```bash
python run_all_4_nasc.py --enable_cascade_on_encoders true --output_suffix cascadeon
python run_all_4_nasc.py --enable_cascade_on_encoders false --output_suffix cascadeoff
```

### Window/Block/TopK Ablation

```bash
for w in 16 32 64 128; do
    python run_all_4_nasc.py --window_size $w --output_suffix w$w
done

for b in 16 32 64; do
    python run_all_4_nasc.py --block_size $b --output_suffix b$b
done

for k in 1 2 3 4; do
    python run_all_4_nasc.py --top_k $k --output_suffix k$k
done
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Path |
|---|---|---|
| DistilBERT | 90.48% ± 2% | Dense |
| RoBERTa | 89.22% ± 3% | Dense |
| GPT-2 | 85.32% ± 2% | Cascade |

## 12.7 Diagnostics

### Verify Cascade Path

Add to `forward`:

```python
logger.info(f"Path: {'dense' if use_dense else 'cascade'}")
```

### Verify Gate

```python
gates = F.softmax(model.gate_proj(summary), dim=-1)
print(f"Gate weights: {gates.mean(dim=0)}")
# Should sum to 1
```

### Verify RFF Memory

```python
KV_cumsum_size = B * H * L * 2 * F * d
print(f"KV_cumsum elements: {KV_cumsum_size:,}")
print(f"Memory (FP32): {KV_cumsum_size * 4 / 1e6:.1f} MB")
```

## 12.8 Known Issues

- **Encoder defaults to dense.** Cascade not exercised.
- **Batch-level gate.** No per-token adaptivity.
- **RFF cumsum OOM** at $L > 512$.
- **Sparse block Python loop.**
- **`inject_nasc` raises on failure.**

## 12.9 Data Availability

- Code: `framework/attentions/nasc.py`
- Metrics: `results/unified_nasc/*/summary.csv`
- Logs: `benchmark_logs/nasc_*.log`

---