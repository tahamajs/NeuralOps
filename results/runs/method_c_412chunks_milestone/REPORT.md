# Method C (412 Chunks, Constant LR): Milestone Champion Run

## Configuration
- **Module:** `mamba_block` (`d_state=16`)
- **Method:** `C` (Curriculum)
- **Dataset Size:** 412 chunks (from 500 texts, fresh teacher cache `teacher_dca952ea154c`)
- **Steps:** 1000
- **LR Schedule:** Constant LR ($2.5 \times 10^{-5}$) after 100-step linear warmup
- **Batch Size:** 8 (effective batch: 32 with accum=4)

## Performance
- **Pre-KD PPL:** 4585.77
- **Post-KD PPL (Step 1000):** **51.18** 🏆
- **Recovery:** **99.57%**
- **Ratio to Baseline (31.54):** **1.62x**
- **Training Time:** 978.62s (~16.3 mins, 978.6 ms/step)
- **Peak VRAM:** 5173.37 MB
- **System Recommendation:** `"YES: SSM can learn from attention. Distillation successfully bridged the architectural gap."`

## Trajectory
- Step 100: 1076.99 (34.1x)
- Step 200: 338.48 (10.7x)
- Step 300: 203.94 (6.47x)
- Step 400: 141.13 (4.47x)
- Step 500: 107.63 (3.41x)
- Step 600: 84.97 (2.69x)
- Step 700: 72.56 (2.30x)
- Step 800: 63.39 (2.01x)
- Step 900: 56.42 (1.79x)
- Step 1000: **51.18** (**1.62x**)

## Conclusion
- Expanding the dataset by 4.4x broke through the 138 PPL ceiling, confirming that data diversity was the decisive bottleneck.
