# Cross-Dataset Transfer

## Purpose

Test whether a compressed model generalizes.

## Evaluations

| Train | Test |
|---|---|
| AG News | SST-2 |
| SST-2 | AG News |
| AG News | MNLI (zero-shot) |
| AG News | CoLA (zero-shot) |

## Metrics

- Accuracy on target dataset
- Drop vs. baseline

## Output

- `papers/modules/<module>/tables/transfer_agnews_to_sst2.csv`
- `papers/modules/<module>/tables/transfer_sst2_to_agnews.csv`
- `papers/modules/<module>/tables/zero_shot.csv`
