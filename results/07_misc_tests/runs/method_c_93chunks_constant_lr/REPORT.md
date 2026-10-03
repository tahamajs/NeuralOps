# Method C (93 Chunks, Constant LR): Diagnostic A

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `C` (Curriculum)
- **Dataset Size:** 93 chunks (from 150 texts)
- **Steps:** 500
- **LR Schedule:** Constant LR ($2.5 \times 10^{-5}$) after 100-step linear warmup (`--no-cosine-lr`)
- **Batch Size:** 8 (effective batch: 32 with accum=4)

## Performance
- **Pre-KD PPL:** 4811.35
- **Post-KD PPL (Step 500):** 152.47
- **Recovery:** 97.47%
- **Ratio to Baseline (31.54):** 4.83x
- **Training Time:** 509.01s (~8.5 mins, 1018.0 ms/step)
- **Peak VRAM:** 5165.23 MB

## Comparison at Step 500
- Cosine LR: 161.96
- Constant LR: 152.47 (-9.49 PPL, ~6% relative improvement)

## Conclusion
- LR decay was not the primary plateau cause, ruling out schedule choking.
