# Skill — Jamba2

## Purpose
Hybrid SVD-Mamba + MHA + MoE.

## Rank argument
`rank`

## Known failure
- Three bugs: early return in nested path, output_scale drift, SimpleSSM random
- OOM on 12-layer models at batch > 8

## Special handling
- `03-causal-vs-bidirectional.md`
- `05-parameter-count-verify.md` (adds 3 branches)
- `07-oom-early-warning.md` (mandatory)

## Skip on
DistilBERT, RoBERTa (causal Mamba + bidirectional encoder)
