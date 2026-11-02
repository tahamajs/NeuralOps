# Surgery Pipeline (`neuralops.surgery.pipeline`)

The Surgery Pipeline orchestrates multi-stage model transformation workflows, ensuring strict validation and state rollback capabilities.

## Usage Example

```python
from neuralops.surgery.pipeline import SurgeryRecipe, SurgeryOp

# Construct a composite recipe
recipe = SurgeryRecipe(
    ops=[
        SurgeryOp(name="replace_attention", params={"attention": "chunk_gla"}),
        SurgeryOp(name="shrink_mlp", params={"ratio": 0.8}),
    ],
    name="fast-transformer",
    base_model="meta-llama/Llama-3.1-8B",
)

# Pre-surgery validation
report = recipe.validate(model)
if report.is_safe:
    # Execute with automatic failure rollback
    optimized_model, apply_report = recipe.apply(model, rollback_on_failure=True)
    print(f"Surgery complete: {apply_report.applied_ops}")
```
