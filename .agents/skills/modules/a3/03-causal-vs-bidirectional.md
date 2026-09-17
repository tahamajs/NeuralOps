# Causal vs Bidirectional Check — a3

## Masking & Recurrence Rules

Compatible with all transformer backbones. In Conv1D layers (GPT-2 c_attn, c_proj), weights are transposed prior to SVD.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.
