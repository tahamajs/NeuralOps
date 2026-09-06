# FLAP: Fluctuation-based Adaptive Structured Pruning

**Registered as `flap`**  
*Structured pruning metric utilizing feature fluctuation and weight importance without requiring extensive retraining.*

---

## Overview

FLAP (Fluctuation-based Adaptive Structured Pruning) is a structured pruning method designed to eliminate redundant channels, attention heads, and hidden dimensions from transformer models. Instead of relying solely on first-order weight magnitudes, FLAP evaluates input feature variance (fluctuation) combined with weight norms (e.g. WIFV - Weight-Importance Feature-Variance metric) to quantify the informational contribution of each projection slice.

---

## Methodology & Formulations

Given input activation representation  \in \mathbb{R}^{B 	imes L 	imes d_{	ext{in}}}$ and weight matrix  \in \mathbb{R}^{d_{	ext{out}} 	imes d_{	ext{in}}}$:

### Feature Fluctuation & Metric Computation
1. Feature variation across tokens:
   6815\sigma_j^2 = \operatorname{Var}(X_{:, :, j})6815
2. Weight importance metric (WIFV):
   6815S_j = \|W_{:, j}\|_2 \cdot \sigma_j6815
3. Structured pruning masks: Columns or rows with the smallest importance scores are pruned according to the user-specified sparsity ratio.

---
