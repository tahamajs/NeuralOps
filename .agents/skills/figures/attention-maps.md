# Attention Map Visualization

## Purpose

Show how attention patterns change after compression.

## Sample selection

5 fixed samples from validation set, stored in `papers/_shared/sample_ids.json`.

## Plot per (sample, layer)

- Baseline attention map (B × H × L × L)
- Compressed attention map
- Difference map (KL divergence per query)

## Output

- `papers/modules/<module>/figures/attn_maps/<sample_id>_baseline.png`
- `papers/modules/<module>/figures/attn_maps/<sample_id>_compressed.png`
- `papers/modules/<module>/figures/attn_maps/<sample_id>_diff.png`
