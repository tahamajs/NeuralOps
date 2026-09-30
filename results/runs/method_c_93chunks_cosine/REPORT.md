# Method C (93 Chunks, Cosine LR): Multi-Stage Curriculum Distillation

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `C` (Curriculum: Hidden MSE early → KL Logits late + CE loss)
- **Dataset Size:** 93 chunks (from 150 texts)
- **Steps:** 1000
- **LR Schedule:** Cosine decay with 100-step linear warmup
- **Batch Size:** 8 (effective batch: 32 with accum=4)

## Performance
- **Pre-KD PPL:** 4203.45
- **Post-KD PPL:** 138.52
- **Recovery:** 97.44%
- **Ratio to Baseline (31.54):** 4.39x
- **Training Time:** 1085.33s (~18 mins, 1085.3 ms/step)
- **Peak VRAM:** 5165.48 MB

## Trajectory
- Step 100: 1186.01
- Step 200: 340.10
- Step 300: 224.17
- Step 400: 183.25
- Step 500: 161.96
- Step 600: 151.61
- Step 700: 145.42
- Step 800: 141.78
- Step 900: 139.80
- Step 1000: 138.52

## Observations
- Massive win over Methods A, B, and D.
- Plateaued around step ~700–1000 at ~138 PPL due to token distribution memorization on 93 chunks.
