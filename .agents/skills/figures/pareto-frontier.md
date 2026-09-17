# Pareto Frontier Figure

## Purpose

Show the trade-off between compression ratio and accuracy.

## X-axis

Compression ratio (params_before / params_after)

## Y-axis

Accuracy (%)

## Implementation

```python
def pareto_front(points):
    """points: list of (ratio, accuracy)."""
    points = sorted(points, key=lambda p: p[0])
    front = []
    best = -1
    for ratio, acc in points:
        if acc > best:
            front.append((ratio, acc))
            best = acc
    return front
```

## Output

- `papers/modules/<module>/figures/pareto_frontier.png`
- `papers/modules/<module>/metrics/pareto.json`
