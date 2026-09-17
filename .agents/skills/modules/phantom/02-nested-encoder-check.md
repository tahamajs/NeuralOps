# Nested Encoder Check — phantom

## Architecture Handling

Bypasses out_proj when nested encoder detected (attention.self vs attention.output).

- Flat architecture (DistilBERT): Projections at module root.
- Nested architecture (RoBERTa/BERT): Projections inside `attention.self`.
