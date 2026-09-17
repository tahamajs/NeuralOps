# Parameter Count Verification — zerokv_neo

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 66955010 | 0.0% | PASS (75.92%) |
| RoBERTa | 124647170 | 124647170 | 0.0% | PASS_LOW (50.92%) |
| GPT-2 | 124441344 | 124441344 | 0.0% | PASS_LOW (51.15%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
