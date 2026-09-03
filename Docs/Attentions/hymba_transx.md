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

## Implementation status

The implementation for this page is [hymba_transx.py](./hymba_transx.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
