# 10. Limitations and Failure Modes

We identify two key limitations:
1. **Extremely Low Ranks (r < 16):** Subspace collapse occurs when rank budgets are constrained below the intrinsic dimensionality of attention heads.
2. **Zero-Shot Transfer:** Without brief gradient calibration (100–250 steps), deep models exhibit compounding error accumulation.
