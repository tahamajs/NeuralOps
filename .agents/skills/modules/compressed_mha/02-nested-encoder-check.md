# Nested Encoder Check — compressed_mha

## Architecture Handling

Checks out_lin, c_proj, and o_proj attributes first to ensure DistilBERT flat attention is not misidentified as nested RoBERTa/BERT.

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.
