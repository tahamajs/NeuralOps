# Paper Specific Sections — decay_linear_attention

## Methodology Description

Decay Linear Attention: Recurrent Attention with Exponential Gating
Source implementation: `Codes/framework/attentions/decay_linear_attention.py`.

Mathematical formulation:
$$
S_t = \lambda_t S_{t-1} + K_t^\top V_t, \quad O_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) \sum \lambda_t K_t^\top}
$$

## Results Narrative

- DistilBERT: 62.27% accuracy (PASS_NO_COMPRESSION), +0.96% parameter change.
- RoBERTa: 49.2% accuracy (PASS_NO_COMPRESSION), +6.72% parameter change.
- GPT-2: 49.43% accuracy (PASS_NO_COMPRESSION), +1.04% parameter change.
