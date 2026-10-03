<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 12_reproducibility.md
     PURPOSE: Reproducibility
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/your-org/SVD_Project
cd SVD_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 12.2 Running

```bash
python run_all_4_cosformer.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 16
```

## 12.3 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"Hooshaai/svd-linear-attention-{model}-cosformer",
        filename="weights.pt",
    )
```

## 12.4 Reproducing Tables

### Table 7.1

```bash
python run_all_4_cosformer.py --models distilbert,roberta,gpt2
```

### Table 7.4 (Chunk size)

```bash
for cs in 16 32 64 128; do
    python run_all_4_cosformer.py --chunk_size $cs --output_suffix cs$cs
done
```

### Table 7.5 (Activation)

```bash
python run_all_4_cosformer.py --act_fun relu --output_suffix relu
python run_all_4_cosformer.py --act_fun softplus --output_suffix softplus
```

## 12.5 Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Accuracy | Status |
|---|---|---|
| DistilBERT | 77.75% ± 2% | PASS |
| RoBERTa | 58.72% ± 3% | PASS_LOW |
| GPT-2 | 70.18% ± 2% | PASS |

## 12.7 Diagnostics

### Verify Cos/Sin Reweighting

```python
import math

N = 128
max_L = 512
theta = [math.pi * i / (2 * max_L) for i in range(N)]
assert theta[0] == 0.0
assert theta[N - 1] < math.pi / 2
```

### Verify Causality

```python
x = torch.randn(1, 128, 768, requires_grad=True)
y = model(x)
y[0, 50].sum().backward()
assert x.grad[0, 51:].abs().max() < 1e-6
```

## 12.8 Known Issues

- **RoBERTa instability**: Default hyperparameters cause drop.
- **Slow decoder**: chunked scan adds overhead.
- **Fragile RoPE extraction**: Triple-nested try/except.
- **Dead `tau` parameter**.

## 12.9 Data Availability

- Code: `framework/attentions/cosformer.py`
- Metrics: `results/unified_cosformer/*/summary.csv`
- Logs: `benchmark_logs/cosformer_*.log`

---