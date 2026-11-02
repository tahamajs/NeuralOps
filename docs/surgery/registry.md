# Model Surgery Registry (`neuralops.surgery.registry`)

The `SURGERY_REGISTRY` manages all verified architectural transformation operations for deep learning models within NeuralOps.

## Key APIs

```python
from neuralops.surgery import (
    SURGERY_REGISTRY,
    register_surgery,
    list_surgeries,
    get_surgery,
    get_surgeries_for_model,
)

# List registered surgeries
surgeries = list_surgeries()
print(surgeries)

# Register a custom surgery
@register_surgery(
    name="custom_op",
    category="attention",
    compat=["llama", "qwen", "gpt2"],
    requires_gpu=False,
)
def custom_op(model, **kwargs):
    # Transformation logic
    return model
```

## Metadata Contract
Every surgery operator provides:
- `name`: Unique operator identifier
- `category`: Functional target (`attention`, `mlp`, `layers`, `weights`, `recovery`)
- `compat`: List of validated target architecture families
- `requires_gpu`: Strict hardware requirement flag
- `requires_calibration`: Whether activation calibration datasets are required
- `reversibility`: Estimation of state reversibility (`high`, `medium`, `low`, `none`)
- `cost_estimate`: Estimated execution latency (`low`, `medium`, `high`)
- `min_model_size_b`: Minimum recommended model parameter size in billions
- `_experimental`: Guard flag emitting warnings when external verification is pending
