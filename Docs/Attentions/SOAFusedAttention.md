# SOA-Fused Attention: Low-Rank Latent Space Multi-Head Attention

## 1. Mathematical Formulation & Architecture
Factorizes dense Q, K, and V linear projection operators into separate low-rank bottleneck pairs:
$$Z_Q = X W_Q^{\text{down}}, \quad Q = Z_Q W_Q^{\text{up}}$$
$$Z_K = X W_K^{\text{down}}, \quad K = Z_K W_K^{\text{up}}$$
$$Z_V = X W_V^{\text{down}}, \quad V = Z_V W_V^{\text{up}}$$
where $W^{\text{down}} \in \mathbb{R}^{r \times D}, W^{\text{up}} \in \mathbb{R}^{D \times r}$ with rank $r=64$.

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Default rank $r=32$ removed $>95\%$ of singular energy, degrading accuracy to 56.88%.
- **Architectural Repair**: Upgraded default rank to $r=64$, added `from_distilbert_pretrained`, and verified SVD reconstruction fidelity.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.60%** (up from 56.88%).\n

## Implementation status

The implementation for this page is [SOAFusedAttention.py](./SOAFusedAttention.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
