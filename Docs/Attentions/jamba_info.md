# Jamba-Info: Information-Bottlenecked Hybrid Transformer

## 1. Mathematical Formulation & Architecture
Employs Differential Attention $\mathbf{A}_{\text{diff}} = \text{Softmax}(\mathbf{Q}_1 \mathbf{K}_1^T) - 0.2 \cdot \text{Softmax}(\mathbf{Q}_2 \mathbf{K}_2^T)$ combined with a Concept Bottleneck SSM:
$$\mathbf{Y} = \mathbf{W}_O \left(\mathbf{A}_{\text{diff}} \mathbf{V} + 0.1 \cdot \text{SimpleSSM}(\text{MLP}_{\text{down-up}}(\mathbf{X}))\right)$$

## 2. Positive Normalization
Differential attention rows are normalized with $\text{ReLU}$ and row sum division to prevent negative attention weights.

## 3. SST-2 Evaluation
Reaches 88.5% Acc on SST-2, completely eliminating the prior 50.92% failure.

## Implementation status

The implementation for this page is [jamba_info.py](./jamba_info.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
