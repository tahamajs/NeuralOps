# Skill — SVD-Linear Attention

## Purpose
Canonical low-rank factorized linear attention.

## Rank argument
`rank`

## Known failure
- `output_scale` drift if learning rate too high
- Sequence length scaling behavior on causal vs bidirectional
