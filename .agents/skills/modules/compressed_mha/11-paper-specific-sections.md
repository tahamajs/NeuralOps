# Paper Specific Sections — compressed_mha

## Methodology Description

Compressed MHA: Joint Truncated SVD Multi-Head Attention
Source implementation: `Codes/framework/attentions/compressed_mha_attention.py`.

Mathematical formulation:
$$
W_{qkv} = \begin{bmatrix} W_q \\ W_k \\ W_v \end{bmatrix} \approx U_r \Sigma_r V_r^\top, \quad W_o \approx U_{ro} \Sigma_{ro} V_{ro}^\top
$$

## Results Narrative

- DistilBERT: 50.92% accuracy (PASS_LOW), -19.16% parameter change.
- RoBERTa: 49.89% accuracy (PASS_LOW), -14.90% parameter change.
- GPT-2: 50.92% accuracy (PASS_LOW), -20.62% parameter change.
