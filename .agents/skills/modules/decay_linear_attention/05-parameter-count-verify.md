# Parameter Count Verification — decay_linear_attention

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 67600274 | +0.96% | PASS_NO_COMPRESSION (62.27%) |
| RoBERTa | 124647170 | 133024802 | +6.72% | PASS_NO_COMPRESSION (49.2%) |
| GPT-2 | 124441344 | 125731872 | +1.04% | PASS_NO_COMPRESSION (49.43%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
