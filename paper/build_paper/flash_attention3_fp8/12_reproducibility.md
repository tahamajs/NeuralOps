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
source .venv/bin/activate  # or .venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Verify Hardware First

```python
import torch

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Float8 dtype:", hasattr(torch, "float8_e4m3fn"))
try:
    import flash_attn

    print("flash_attn:", flash_attn.__version__)
except ImportError:
    print("flash_attn: NOT INSTALLED")
```

Expected on RTX 4070 Laptop:

```
CUDA: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Capability: (8, 9)
Float8 dtype: True
flash_attn: NOT INSTALLED  (or a version without FP8 support)
```

## 12.3 Running

```bash
python run_all_4_flash3_fp8.py \
    --models distilbert,roberta,gpt2 \
    --recovery_steps 50 \
    --lora_rank 8
```

## 12.4 Downloading Checkpoints

```python
from huggingface_hub import hf_hub_download

for model in ["distilbert", "roberta", "gpt2"]:
    path = hf_hub_download(
        repo_id=f"efficiency-lab/svd-linear-attention-{model}-flash3_fp8",
        filename="weights.pt",
    )
    print(f"Downloaded {model}")
```

## 12.5 Reproducing Tables

### Table 7.2 (Main Results)

```bash
python run_all_4_flash3_fp8.py --models distilbert,roberta,gpt2
```

### Table 7.5 (use_fp8 Ablation)

```bash
python run_all_4_flash3_fp8.py --use_fp8 true  --output_suffix fp8on
python run_all_4_flash3_fp8.py --use_fp8 false --output_suffix fp8off
```

## 12.6 Seeds

```python
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Accuracy | FP8 Path |
|---|---|---|
| DistilBERT | 89.79% ± 1% | No |
| RoBERTa | 78.2% ± 1% | No |
| GPT-2 | 77.5% ± 1% | No |

## 12.8 Hardware Requirements for FP8

To reproduce the FP8 path, you need:

| Requirement | Minimum |
|---|---|
| GPU | H100, H200, or RTX 4090 |
| CUDA | 12.0+ |
| PyTorch | 2.1+ |
| `flash-attn` | 2.6+ (Hopper) or `flash-attn-3` |

## 12.9 Known Issues

- Windows: `cp1256` codec issues; set `PYTHONIOENCODING=utf-8`.
- CUDA: FP8 requires CUDA >= 11.8.
- Driver: consumer Ada GPUs may not expose FP8 via driver.

## 12.10 Reproducibility Checklist

- [ ] Hardware printed via `torch.cuda.get_device_name(0)`
- [ ] `hasattr(torch, "float8_e4m3fn")` result logged
- [ ] `flash_attn` import attempt logged
- [ ] FP8 path taken or not taken logged
- [ ] SDPA backend logged

## 12.11 Data Availability

- Code: `framework/attentions/flash_attention3_fp8.py`
- Checkpoints: `efficiency-lab/svd-linear-attention-*-flash3_fp8`
- Metrics: `results/unified_flash3_fp8/*/summary.csv`
- Logs: `benchmark_logs/flash3_fp8_*.log`

---