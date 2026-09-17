# Parameter Count Verification — a3

## Empirical Parameter Counts from Benchmark Runs

| Backbone | Baseline Params | Compressed Params | Change % | Empirical Status |
|---|---|---|---|---|
| DistilBERT | 66955010 | 79525634 | +18.77% | PASS_NO_COMPRESSION (90.14%) |
| RoBERTa | 124647170 | 150871298 | +21.04% | PASS_NO_COMPRESSION (81.19%) |
| GPT-2 | 124441344 | 148060416 | +18.98% | PASS_NO_COMPRESSION (49.08%) |

## Reporting Protocol

If `change_pct >= 0.0%`:
- Report truthfully as architectural replacement / KV pruning / throughput optimization.
- Do NOT falsely label as weight parameter reduction.
