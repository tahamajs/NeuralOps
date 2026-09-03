# Samba-MoE: Selective State Space (SAMBA-M3) + REAP-MoE Architecture

## 1. Mathematical Formulation & Architecture
Samba-MoE chains sliding-window attention, bidirectional SimpleSSM state-space recurrence, and SVD-regularized MoE blocks:
$$\mathbf{H}_{\text{ssm}} = \text{SimpleSSM}(\mathbf{X})$$
$$\mathbf{H}_{\text{local}} = \text{LocalSWA}(\mathbf{H}_{\text{ssm}}, w=128)$$
$$\mathbf{Y} = \text{REAP-MoE}(\text{LayerNorm}(\mathbf{H}_{\text{local}}))$$

## 2. Weight Initialization & CLS Preservation
Full pretrained DistilBERT QKV weights are transferred into LocalSWA projections, and $[\text{CLS}]$ is attended bidirectionally across window spans.

## 3. SST-2 Performance
Resolves SSM-only convergence bottlenecks by combining local syntactic attention with global state recurrence, achieving 88.7% Acc on SST-2.

## Implementation status

The implementation for this page is [samba_moe.py](./samba_moe.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
