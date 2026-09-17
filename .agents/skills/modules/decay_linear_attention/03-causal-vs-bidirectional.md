# Causal vs Bidirectional Check — decay_linear_attention

## Masking & Recurrence Rules

Forward-only linear recurrence enforces causal masking for autoregressive decoders. Bidirectional recurrence enabled for encoders.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.
