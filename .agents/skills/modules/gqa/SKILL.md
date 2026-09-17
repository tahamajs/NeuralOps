# Skill — Grouped Query Attention

## Purpose
Reduce KV head count from H to H_kv.

## Rank argument
`num_kv_heads`

## Known failure
- On MHA models (DistilBERT, RoBERTa, GPT-2), forcing H_kv < H drops information
- Silent truncation of K/V weights in copy_weights

## Special handling
- `05-parameter-count-verify.md` (GQA shouldn't reduce params on MHA)
- `10-honest-reporting.md` (report as "re-architecture")

## Skip on
DistilBERT, RoBERTa, GPT-2 (they are MHA, not GQA)
