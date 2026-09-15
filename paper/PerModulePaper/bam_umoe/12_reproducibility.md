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

## 12.3 Running BAM-UMoE

```bash
python run_all_4_bam_umoe.py \
    --models distilbert,roberta,gpt2 \
    --num_experts {{NUM_EXPERTS}} \
    --top_k {{TOP_K}} \
    --lora_rank {{LORA_RANK}} \
    --recovery_steps {{RECOVERY_STEPS}}
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
| DistilBERT | {{D_ACC}}% ± 1% | {{D_PARAMS}}M |
| RoBERTa | {{R_ACC}}% ± 1% | {{R_PARAMS}}M |
| GPT-2 | {{G_ACC}}% ± 1% | {{G_PARAMS}}M |

## 12.7 Data Availability

- Dataset: SST-2 via `datasets`
- Checkpoints: `{{HF_ORG}}/svd-linear-attention-*-bam_umoe`
- Metrics: `results/unified_bam_umoe/*/summary.csv`

## 12.8 Known Issues

- Windows: `PYTHONIOENCODING=utf-8`
- CUDA ≥ 11.8 for FP16
- Large parameter count may OOM on small GPUs
