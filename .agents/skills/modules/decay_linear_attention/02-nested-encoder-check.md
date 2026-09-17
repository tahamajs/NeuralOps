# Nested Encoder Check — decay_linear_attention

## Architecture Handling

Nested encoder detection in place to bypass out_proj in inner self-attention blocks.

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.
