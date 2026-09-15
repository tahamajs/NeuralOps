# 7. Results

## 7.1 Main Results

| Model | Baseline Acc. | Compressed Acc. | Delta | Parameter Reduction | VRAM Reduction |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 90.7% | 88.5% | -2.2% | ~60% | ~18% |
| RoBERTa-base | 91.1% | 89.0% | -2.1% | ~68% | ~25% |
| GPT-2 | 90.3% | 88.4% | -1.9% | ~62% | ~30% |

## 7.2 Ablations

We vary rank, calibration size, and the intensity of the LayerNorm recalibration to understand how much of the loss is due to approximation error versus distribution shift.

## 7.3 Discussion of Gains

The largest gains come from compressing the linear projections that dominate the model footprint. The activation-aware weighting preserves the most important directions and prevents the compressed model from collapsing onto low-energy subspaces.
