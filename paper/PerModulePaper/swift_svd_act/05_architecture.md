# 5. Architecture and Implementation

## 5.1 Submodule Structure
The core computation of `swift_svd_act` is encapsulated within `SwiftSVDActivationCompressor`. The module exposes standardized forward interfaces:
- Base Weight Tensor: Frozen or structured parameter buffer
- Adapter Branches: Low-rank update components (when configured)
- Conv1D Transposition Compatibility: Explicit handling for autoregressive heads (GPT-2)
- Multi-Head Attention Alignment: Preservation of Key/Value projections in Grouped-Query Attention (GQA) architectures.
