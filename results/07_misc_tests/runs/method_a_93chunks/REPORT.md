# Method A: Logits-Only Knowledge Distillation (KL Divergence)

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `A` (KL Divergence with T=2.0 + Cross-Entropy)
- **Dataset Size:** 93 chunks (from 150 texts)
- **Steps:** 300 (initial sweep) / 1000
- **LR Schedule:** Cosine decay (base: 5e-6, injected: 2.5e-5)
- **Batch Size:** 2 (effective batch: 8 with accum=4)

## Performance
- **Pre-KD PPL:** 7302.55
- **Post-KD PPL:** 1068.77
- **Recovery:** 85.73%
- **Ratio to Baseline (31.54):** 33.88x
- **Training Time:** 225.15s (750.5 ms/step)
- **Peak VRAM:** 2371.16 MB

## Observations
- Directly enforcing output logit distributions without representation alignment led to high gradient variance and loss oscillation.
- Post-KD PPL stalled above 1000.
