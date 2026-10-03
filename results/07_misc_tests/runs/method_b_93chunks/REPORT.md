# Method B: Hidden State Feature Alignment (MSE Loss)

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `B` (Mean Squared Error on all 12 layer hidden states + Learnable Linear Projection Adapter + Cross-Entropy)
- **Dataset Size:** 93 chunks (from 150 texts)
- **Steps:** 300 (sweep) / 1000
- **LR Schedule:** Cosine decay
- **Batch Size:** 2 (effective batch: 8 with accum=4)

## Performance
- **Pre-KD PPL:** 5269.20
- **Post-KD PPL:** 775.22
- **Recovery:** 85.80%
- **Ratio to Baseline (31.54):** 24.58x
- **Training Time:** 242.19s (807.3 ms/step)
- **Peak VRAM:** 2386.62 MB

## Observations
- Captures intermediate manifold geometry significantly better than Method A (775 vs 1068).
- Without output logit distillation (KL), final vocabulary ranking is suboptimal.
