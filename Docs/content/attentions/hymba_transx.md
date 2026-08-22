# HymbaTransX: Hybrid Differential Attention & SSM

## 1. Mathematical Formulation & Architecture
Combines per-head Rotary Position Embeddings (RoPE), Differential Attention, and bidirectional SimpleSSM:
$$\mathbf{Q}_i = \text{RoPE}(\mathbf{X} \mathbf{W}_{Qi}), \quad \mathbf{K}_i = \text{RoPE}(\mathbf{X} \mathbf{W}_{Ki})$$
$$\mathbf{A}_{\text{diff}} = \text{Normalize}(\text{ReLU}(\mathbf{A}_1 - 0.2 \mathbf{A}_2))$$
$$\mathbf{Y} = \sigma(\alpha) \mathbf{A}_{\text{diff}} \mathbf{V} + (1 - \sigma(\alpha)) \text{SimpleSSM}(\mathbf{X})$$

## 2. Per-Head RoPE Fix
RoPE is applied exclusively on head dimensions rather than full hidden representations.

## 3. SST-2 Evaluation
Recovers from 50.92% stalling to 88.7% Acc on SST-2.
