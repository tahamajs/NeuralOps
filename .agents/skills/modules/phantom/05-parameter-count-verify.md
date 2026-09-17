# Parameter Count Verification — phantom

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 54568730 | -18.50% | PASS (72.59%) |
| RoBERTa | 124647170 | 106961714 | -14.19% | PASS_LOW (50.92%) |
| GPT-2 | 124441344 | 99668784 | -19.91% | PASS_LOW (50.92%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
