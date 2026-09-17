# Skill — SVD-Mamba

## Purpose
SVD-factorized selective state space model.

## Rank argument
`rank`

## Known failure
- SSM requires causal sequence — incompatible with bidirectional encoders
- `output_scale` drifts during recovery
- `x_proj` and `dt_proj` randomly initialized

## Special handling
- `03-causal-vs-bidirectional.md` (mandatory skip on BERT/RoBERTa)
- `01-copy-weights-audit.md` (verify q/k/v)
- `06-forward-sanity-check.md`

## Skip on
DistilBERT, RoBERTa (causal-only)
