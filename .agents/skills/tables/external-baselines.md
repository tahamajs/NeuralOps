# External Baselines Table

## Required baselines

At minimum, reproduce:

| Baseline | Reference |
|---|---|
| SVD (Eckart-Young) | Eckart & Young 1936 |
| LoRA | Hu et al. 2021 |
| SparseGPT (if pruning-relevant) | Frantar et al. 2023 |
| AWQ (if quantization-relevant) | Lin et al. 2023 |
| H2O (if KV-cache-relevant) | Zhang et al. 2023 |
| StreamingLLM (if KV-cache-relevant) | Xiao et al. 2023 |

## Columns

| Method | Backbone | Rank / Setting | Acc | Params ratio | Source |
|---|---|---|---|---|---|

## Output

- `papers/modules/<module>/tables/external_baselines.csv`
- Cited in `paper-writing/related-work-template.md`
