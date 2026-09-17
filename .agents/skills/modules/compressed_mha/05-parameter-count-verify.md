# Parameter Count Verification — compressed_mha

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 54128038 | -19.16% | PASS_LOW (50.92%) |
| RoBERTa | 124647170 | 106079074 | -14.90% | PASS_LOW (49.89%) |
| GPT-2 | 124441344 | 98786144 | -20.62% | PASS_LOW (50.92%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
