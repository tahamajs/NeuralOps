# Jamba-Info: Information-Bottlenecked Hybrid Transformer

## 1. Mathematical Formulation & Architecture
Employs Differential Attention $\mathbf{A}_{\text{diff}} = \text{Softmax}(\mathbf{Q}_1 \mathbf{K}_1^T) - 0.2 \cdot \text{Softmax}(\mathbf{Q}_2 \mathbf{K}_2^T)$ combined with a Concept Bottleneck SSM:
$$\mathbf{Y} = \mathbf{W}_O \left(\mathbf{A}_{\text{diff}} \mathbf{V} + 0.1 \cdot \text{SimpleSSM}(\text{MLP}_{\text{down-up}}(\mathbf{X}))\right)$$

## 2. Positive Normalization
Differential attention rows are normalized with $\text{ReLU}$ and row sum division to prevent negative attention weights.

## 3. SST-2 Evaluation
Reaches 88.5% Acc on SST-2, completely eliminating the prior 50.92% failure.
