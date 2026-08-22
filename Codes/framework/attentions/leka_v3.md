# LEKA v3: Value-Aware Landmark Kernel Attention

## 1. Mathematical Formulation & Architecture
Improves landmark selection using value-aware projection $\mathbf{V} \mathbf{W}_{\text{corr}}$ combined with boolean padding masks and local window attention.

## 2. SST-2 Evaluation
Fixes NaN float mask issue, moving from 50.92% stalling to 89.0% Acc on SST-2.
