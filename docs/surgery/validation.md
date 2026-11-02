# Pre-Surgery Validation (`neuralops.surgery.validation`)

Validation gates execute prior to modifying production weights, ensuring structural and numerical integrity.

## The 5 Validation Checks

1. `architecture_compat`: Verifies every operator supports the target model family.
2. `recipe_compat`: Detects conflicting or redundant modifications in the pipeline.
3. `forward_pass`: Tests single-batch model execution before mutating weights.
4. `output_sanity`: Checks output tensors for NaN, Inf, and abnormal norm explosions.
5. `quality_probe`: Measures baseline loss and perplexity delta against acceptable quality budgets.

## API Usage

```python
from neuralops.surgery.validation import validate_before_surgery

report = validate_before_surgery(model, recipe, quality_budget=0.02)
assert report.is_safe, f"Validation failed: {report.warnings}"
```
