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

## 12.3 Running BAM-UMoE

```bash
python run_all_4_bam_umoe.py \
    --models distilbert,roberta,gpt2 \
    --num_experts 8 \
    --top_k 2 \
    --lora_rank 16 \
    --recovery_steps 50
```

## 12.4 Reproducing Tables

```bash
# Main results
python run_all_4_bam_umoe.py --models distilbert,roberta,gpt2

# Expert count ablation
for E in 2 4 8 16; do
    python run_all_4_bam_umoe.py --num_experts $E --output_suffix E$E
done

# Top-k ablation
for k in 1 2 4; do
    python run_all_4_bam_umoe.py --top_k $k --output_suffix k$k
done
```

## 12.5 Seeds

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

## 12.6 Expected Results

| Model | Acc | Params |
|---|---|---|
| DistilBERT | 90.02% ± 1% | 293M |
| RoBERTa | 84.63% ± 1% | 570M |
| GPT-2 | 81.42% ± 1% | 570M |

## 12.7 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `Hooshaai/svd-linear-attention-*-bam_umoe`
- Metrics: `results/unified_bam_umoe/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Large parameter count may OOM on small GPUs

---