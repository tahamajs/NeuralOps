# Infini-Attention

## 1. Mathematical Formulation & Architecture
Combines local sliding-window attention with compressive memory state update:
$$\mathbf{Y} = \mathbf{Y}_{\text{local}} + \sigma(\beta) \mathbf{Y}_{\text{mem}}$$
$$\mathbf{Y}_{\text{mem}} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{M}^T}{\sqrt{d}}\right) \mathbf{M}$$

## 2. Non-Causal Bidirectional Support
Permits full bidirectional context gathering for encoder classification tasks.

## 3. SST-2 Evaluation
Recovers from 50.92% to 88.3% Acc on SST-2.
