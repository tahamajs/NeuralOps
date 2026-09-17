# Causal vs Bidirectional Check — aasvd

## Masking & Recurrence Rules

Architecture-agnostic linear layer compression. Works across causal decoders (GPT-2) and bidirectional encoders (RoBERTa, DistilBERT).

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.
