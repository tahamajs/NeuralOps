# Paper Specific Sections — a3

## Methodology Description

A3: Activation-Aware SVD Compression with Fisher Energy Weighting
Source implementation: `Codes/framework/compression_engine/a3_compressor.py`.

Mathematical formulation:
$$
C_X = \frac{1}{N} \sum x_t x_t^\top, \quad F = \mathbb{E}[x^2], \quad W_{weighted} = W \operatorname{diag}(\sqrt{F+\epsilon}) = U\Sigma V^\top
$$

## Results Narrative

- DistilBERT: 90.14% accuracy (PASS_NO_COMPRESSION), +18.77% parameter change.
- RoBERTa: 81.19% accuracy (PASS_NO_COMPRESSION), +21.04% parameter change.
- GPT-2: 49.08% accuracy (PASS_NO_COMPRESSION), +18.98% parameter change.
