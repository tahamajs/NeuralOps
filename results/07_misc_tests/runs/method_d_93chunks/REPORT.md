# Method D: SSM-Specialized Cosine Similarity Distillation

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `D` (Cosine Similarity on Hidden States + KL Divergence with T=4.0)
- **Dataset Size:** 93 chunks (from 150 texts)
- **Steps:** 300 (sweep) / 1000
- **LR Schedule:** Cosine decay
- **Batch Size:** 2 (effective batch: 8 with accum=4)

## Performance
- **Pre-KD PPL:** 5029.02
- **Post-KD PPL:** 868.29
- **Recovery:** 83.26%
- **Ratio to Baseline (31.54):** 27.53x
- **Training Time:** 261.95s (873.2 ms/step)
- **Peak VRAM:** 2457.67 MB

## Observations
- Cosine similarity normalizes vector norms, discarding magnitude information critical for softmax scaling.
- Temperature T=4 amplifies the gradient variance ($T^2 = 16$), underperforming MSE-based alignment.
