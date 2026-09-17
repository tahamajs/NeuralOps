# Parameter Count Verification — aasvd

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 65333005 | -2.42% | PASS (87.96%) |
| RoBERTa | 124647170 | 123172620 | -1.18% | PASS (86.7%) |
| GPT-2 | 124441344 | 112055064 | -9.95% | PASS (69.38%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
