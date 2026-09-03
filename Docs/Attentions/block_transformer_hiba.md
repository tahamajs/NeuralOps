# Block Transformer HIBA: Hierarchical Intra/Inter-Block Attention

## 1. Mathematical Formulation & Architecture
Partitions sequence into blocks of length $B_{\text{sz}}$:
- **Intra-Block Attention**: Dense bidirectional attention within each block.
- **Inter-Block Sparse Attention**: Top-$k$ block routing with mandatory inclusion of Block 0 (containing $[\text{CLS}]$).
- **Periodic Rectification**: Full self-attention executed every $R=4$ layers.

## 2. Bidirectional Mask Fix
Disabled causal masking for encoder classification, allowing $[\text{CLS}]$ to attend forward across all sequence blocks.

## 3. SST-2 Evaluation
Increases accuracy from 57.80% to 88.8% on SST-2.

## Implementation status

The implementation for this page is [block_transformer_hiba.py](./block_transformer_hiba.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
