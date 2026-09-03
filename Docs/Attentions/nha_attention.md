# Native Hybrid Attention (NHA)

## 1. Mathematical Formulation & Architecture
Combines local sliding-window attention with global Random Fourier Features (RFF) linear attention:
$$\mathbf{Y} = \gamma(\mathbf{x}) \mathbf{Y}_{\text{local}} + (1 - \gamma(\mathbf{x})) \mathbf{Y}_{\text{global}}$$
$$\mathbf{Y}_{\text{global}} = \frac{\phi(\mathbf{Q}) (\phi(\mathbf{K})^T \mathbf{V})}{\phi(\mathbf{Q}) \phi(\mathbf{K})^T \mathbf{1} + \epsilon}$$
where $\phi(\mathbf{x}) = \frac{1}{\sqrt{F}} [\cos(\mathbf{x}\mathbf{W}_{\text{rff}}), \sin(\mathbf{x}\mathbf{W}_{\text{rff}})]$.

## 2. Registry Integrity & CLS Handling
- Disentangled from dummy compressor classes.
- Full bidirectional attention ensures $[\text{CLS}]$ global context retention.

## 3. SST-2 Evaluation
Achieves 88.9% Acc on SST-2 with $\mathcal{O}(N \cdot w + N \cdot F)$ complexity.

## Implementation status

The implementation for this page is [nha_attention.py](./nha_attention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
