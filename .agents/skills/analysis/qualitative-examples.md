# Qualitative Examples

## Purpose

Show concrete examples of model behavior.

## Samples

- Top 5 correct predictions (highest confidence)
- Top 5 wrong predictions (highest confidence)
- 5 samples where baseline and compressed disagree

## Format

```json
{
  "sample_id": "agnews_val_0042",
  "text": "...",
  "ground_truth": "Business",
  "baseline_pred": "Business",
  "baseline_conf": 0.92,
  "compressed_pred": "World",
  "compressed_conf": 0.71,
  "notes": "Compressed model confused by economic + geopolitical terms"
}
```

## Output

- `papers/modules/<module>/examples/best_cases.json`
- `papers/modules/<module>/examples/worst_cases.json`
- `papers/modules/<module>/examples/comparison.jsonl`
- `papers/modules/<module>/examples/disagreements.json`
