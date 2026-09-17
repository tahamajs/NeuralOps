# Causal vs Bidirectional Check — zerokv_neo

## Masking & Recurrence Rules

Native support for causal decoders (GPT-2, Qwen) using attention sink tokens. Bidirectional encoders (RoBERTa, DistilBERT) retain sinks and top-k attention scores.

- Causal decoders (GPT-2, Qwen): Must maintain strict autoregressive triangular masking.
- Bidirectional encoders (RoBERTa, DistilBERT): Must allow bidirectional attention context.
