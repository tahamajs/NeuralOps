# FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention

## 1. Mathematical Formulation & Architecture
Combines block-level Seer gating with fine-grained selected attention:
$$\mathbf{S}_{\text{blk}} = \frac{\mathbf{G}_Q(\mathbf{Q}_{\text{blk}}) \mathbf{G}_K(\mathbf{K}_{\text{blk}})^T}{\tau}$$
Block 0 ($[\text{CLS}]$) is permanently retained in top-$k$ block selection.

## 2. SST-2 Evaluation
Reaches 88.6% Acc on SST-2.

## Implementation status

The implementation for this page is [fusion_attn.py](./fusion_attn.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
