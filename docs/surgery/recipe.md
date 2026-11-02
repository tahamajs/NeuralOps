# Surgery Recipes (`neuralops.surgery.recipe`)

Declarative recipes allow reproducible serialization, versioning, and sharing of model optimizations across organizations.

## YAML Format

```yaml
recipe_version: "1.0"
name: llama3-8b-fast
base_model: meta-llama/Llama-3.1-8B
surgeries:
  - name: replace_attention
    params:
      attention: chunk_gla
      layers: "all"
  - name: shrink_mlp
    params:
      ratio: 0.7
recovery:
  - name: distill_from_teacher
    params:
      steps: 500
```

## Serialization APIs

```python
from neuralops.surgery.recipe import save_recipe, load_recipe

# Save recipe
save_recipe(recipe, "recipes/llama3-fast.yaml")

# Load recipe
recipe = load_recipe("recipes/llama3-fast.yaml")
```
