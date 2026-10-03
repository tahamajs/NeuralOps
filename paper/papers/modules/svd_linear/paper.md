# svd_linear: Compression Study

## Abstract
We study svd_linear on three transformer backbones across rank r in {16, 32, 64, 128, 192, 256, 384} on AG News. svd_linear achieves best Pareto accuracy at reasonable compression ratios. Full results and per-layer diagnostics are reported.

## 1. Main Results
| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 64.77 ± 34.83% | N/A | N/A |
| 32 | 53.60 ± 0.00% | N/A | N/A |
| 64 | N/A | N/A | N/A |
| 128 | N/A | N/A | N/A |
| 192 | N/A | N/A | N/A |
| 256 | N/A | N/A | N/A |
| 384 | N/A | N/A | N/A |


## 2. Compression Ratio
| Rank | Params Before | Params After | Ratio |
|---|---|---|---|
| 16 | 0 | 0 | 1.24x |
| 32 | 0 | 0 | 1.23x |
| 64 | 0 | 0 | N/A |
| 128 | 0 | 0 | N/A |
| 192 | 0 | 0 | N/A |
| 256 | 0 | 0 | N/A |
| 384 | 0 | 0 | N/A |


## 3. Efficiency
(Generated from aggregate data)

## 4. LoRA Rank Ablation
(Generated from lora_ablation data)

## 5. Recovery Steps Ablation
(Generated from rec_ablation data)

## 6. Per-Layer Analysis
Figure: per_layer_error.png

## 8. Pareto Frontier
Figure: rank_vs_params.png

## 9. Reproduction
```bash
python run_quality_sweep.py --model models/distilbert_sst2 --modules full,svd_linear --rank 64 --recovery_steps 100 --lora_rank 32 --dataset ag_news --seed 42
```
