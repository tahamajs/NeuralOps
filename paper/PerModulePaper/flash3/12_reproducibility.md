# 12. Reproducibility Statement

## 12.1 Environment

```bash
git clone https://github.com/Efficiency Lab/SVD_Project
cd SVD_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 12.2 Optional: FlashAttention Installation

```bash
# For Hopper (H100/H200):
pip install flash-attn --no-build-isolation

# For older GPUs, no installation needed (SDPA fallback works).
```

## 12.3 Running the Wrapper

```bash
python run_all_4_flash3.py \
    --models distilbert,roberta,gpt2 \
    --use_fp8 True \
    --recovery_steps 50
```

## 12.4 Verifying Path Selection

```python
from framework.attentions.flash_attention3 import FLASH_ATTN_AVAILABLE
print("FlashAttention available:", FLASH_ATTN_AVAILABLE)
```

Expected: `True` on H100, `False` otherwise.

## 12.5 Reproducing Tables

```bash
# Main
python run_all_4_flash3.py --models distilbert,roberta,gpt2

# LoRA rank sweep
for r in 0 4 8 16 32; do
    python run_all_4_flash3.py --lora_rank $r --output_suffix r$r
done
```

## 12.6 Seeds

```python
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.7 Expected Results

| Model | Acc |
|---|---|
| DistilBERT | 89.91% ± 1% |
| RoBERTa | 83.94% ± 1% |
| GPT-2 | 82.00% ± 1% |

## 12.8 Data Availability

- Dataset: SST-2
- Checkpoints: `efficiency-lab/svd-linear-attention-*-flash3`
- Metrics: `results/unified_flash3/*/summary.csv`

## 12.9 Known Issues

- No FA3 on non-Hopper GPUs
- FP8 not tested on Hopper
- Some paths auto-fallback silently (see logs)

---