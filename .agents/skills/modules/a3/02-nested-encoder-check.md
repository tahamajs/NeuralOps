# Nested Encoder Check — a3

## Architecture Handling

Uses should_compress_linear and inspects base_layer. Skips classifier, score, lm_head, pooler layers.

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.
