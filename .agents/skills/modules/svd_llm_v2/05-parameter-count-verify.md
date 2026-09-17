# Parameter Count Verification — svd_llm_v2

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 54569480 | -18.50% | PASS (81.88%) |
| RoBERTa | 124647170 | 106963214 | -14.19% | PASS_LOW (50.92%) |
| GPT-2 | 124441344 | 99670284 | -19.91% | PASS_LOW (50.92%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
