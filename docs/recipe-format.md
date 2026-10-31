# NeuralOps Recipe Format Guide (v1.0.0)

This document provides a practical, user-facing guide to authoring, validating, and executing **NeuralOps Recipe Language (NRL) v1.0.0** recipes.

---

## 1. Overview

A NeuralOps Recipe is a declarative contract that defines a sequence of post-training model optimizations. Unlike arbitrary Python code that mutates model checkpoints imperatively in memory, a recipe can be:
- **Pre-validated** before any device memory is allocated.
- **Audited** against strict architecture and parameter schemas.
- **Hashed** into a canonical SHA-256 fingerprint for publication and verification.
- **Safely rolled back** on failure to preserve model integrity.

---

## 2. Anatomy of a Recipe

A recipe consists of five top-level sections:

```yaml
apiVersion: "neuralops/v1"
kind: "Recipe"

metadata:
  name: "distilbert-fast-aasvd"
  version: "1.0.0"
  description: "Activation-Aware SVD compression on DistilBERT attention layers."
  author: "Taha Majlesi"
  tags: ["distilbert", "aasvd", "compression"]
  license: "MIT"
  production_ready: true

model:
  source: "distilbert-base-uncased"
  dtype: "float32"
  revision: "6cdc0aad91a5ae9d8e7c10b77e8a9fcf45db04a2"
  family: "bert"

operations:
  - id: "compress-attn"
    type: "weights.compress"
    version: "1.0.0"
    config:
      method: "aasvd"
      target_ratio: 0.5
      rank_step: 32
      whitening: true

validation:
  profile: "standard"
  policy: "fail_fast"
  thresholds:
    max_perplexity_increase: 0.05

execution:
  device: "cpu"
  seed: 42
  rollback: true

artifacts:
  output_dir: "./output/distilbert-aasvd"
  save_model: true
  save_report: true
  format: "hf"
```

---

## 3. Section Explanations

### 3.1 Metadata (`metadata`)
Stores provenance and discovery tags:
- `name` (required): DNS-like identifier matching `^[a-z0-9][a-z0-9-]{0,62}$`.
- `version` (required): Strict SemVer 2.0.0 string (e.g., `1.0.0`).
- `description` (required): Human-readable explanation of transformation purpose.
- `author` (optional): Author or team name.
- `tags` (optional): List of search keywords.
- `production_ready` (optional): Set to `false` when testing with `validation.profile: "none"`.
- `references` (optional): List of external URLs or DOI citations.

### 3.2 Model (`model`)
Declares the base model foundation:
- `source` (required): HuggingFace model identifier or local checkpoint path.
- `dtype` (required): Target tensor precision (`bfloat16`, `float16`, `float32`, `fp8`).
- `revision` (optional): 40-character Git commit hash for exact checkpoint pinning. Floating branch names like `main` or `latest` are rejected to guarantee reproducibility.
- `trust_remote_code` (optional): Boolean flag passed to HuggingFace model loader.
- `device_map` (optional): HuggingFace device allocation strategy (e.g., `auto`, `cpu`).
- `family` (optional): Model architecture family (`llama`, `bert`, `gpt2`, `generic`).

### 3.3 Operations (`operations`)
The ordered pipeline of transformations:
- `id` (optional): Stable identifier for dependency tracking.
- `type` (required): Registered operation name (e.g., `weights.compress`, `attention.replace`).
- `version` (required): Pinned semantic version of the operation contract (e.g., `1.0.0`).
- `config` (required): Operation-specific parameter payload, strictly validated against its contract schema.
- `depends_on` (optional): List of preceding operation IDs that must succeed before this stage runs.

### 3.4 Validation (`validation`)
Configures the multi-tier safety gates:
- `profile` (required):
  - `none`: Bypass gates (requires `metadata.production_ready: false`).
  - `structural`: L1 gate only (verifies shapes, dtypes, and tensor connectivity).
  - `numerical`: L1 + L2 gates (verifies forward pass execution without NaN or Inf).
  - `behavioral`: L1 + L2 + L3 gates (measures perplexity delta and output similarity).
  - `empirical`: L1 + L2 + L3 + L4 gates (evaluates on hardware benchmarks and real datasets).
  - `production`: Exhaustive certification.
- `policy` (optional): `fail_fast` (halt on first gate failure), `continue_on_warning`, or `collect_all`.
- `gates` (optional): Explicit gate override list.
- `thresholds` (optional): Numerical criteria overrides (e.g., `max_perplexity_increase: 0.05`).

### 3.5 Execution (`execution`)
Execution runtime parameters:
- `device`: Device placement (`auto`, `cuda`, `cpu`, `mps`).
- `seed`: Integer RNG seed for deterministic transformations.
- `rollback`: Boolean indicating whether to automatically restore snapshots on failure (default: `true`).
- `memory_budget_gb`: Optional peak host/device memory allocation limit.

### 3.6 Artifacts (`artifacts`)
Terminal export configuration:
- `output_dir` (required): Destination folder for transformed checkpoints.
- `save_model` (optional): Whether to serialize model weights (default: `true`).
- `save_report` (optional): Whether to emit structured verification evidence (default: `true`).
- `format`: Export layout (`hf`, `safetensors`, `vllm`, `sglang`, `onnx`, `gguf`).

---

## 4. Canonical Hashing (`recipe_hash`)

Every recipe document can be digested into a unique SHA-256 fingerprint.
The hashing engine:
1. Strips comments and surface formatting.
2. Sorts mapping keys lexicographically by UTF-8 bytes at all depths.
3. Eliminates insignificant whitespace.
4. Normalizes text to Unicode NFC format.

Two recipes differing only by indentation, comments, or key order will compute **identical** recipe hashes.

---

## 5. Well-Formedness Rules Summary

Every recipe must conform to the 15 Well-Formedness Rules defined in NRL v1.0.0:
- **W1 (Root Fields):** `apiVersion`, `kind`, `metadata`, `model`, `operations` must be present.
- **W2 (SemVer):** Version tags must strictly follow `X.Y.Z`.
- **W3 (Registry):** Every operation type must exist in the active registry.
- **W4 (Version Check):** Pinned operation version must match available registry contracts.
- **W5 (Parameter Schema):** Configuration objects must satisfy operation JSON schemas.
- **W6 (Fail-Closed):** Unknown attributes are strictly forbidden at all nesting depths.
- **W7 (Unique IDs):** Operation identifiers must not collide.
- **W8 (Dependency Resolution):** `depends_on` must refer to declared preceding operation identifiers.
- **W9 (Acyclic Order):** Dependencies cannot refer forward in the operation array.
- **W10 (Type Chain):** Lifecycle states must unify cleanly across consecutive transformations.
- **W11 (Preconditions):** Declared model family must satisfy all operation prerequisites.
- **W12 (Self-Composition):** Operations declare whether multiple self-invocations are allowed.
- **W13 (Dataset Requirement):** Profiles `empirical` and `production` require evaluation datasets.
- **W14 (Gate Reference):** Threshold overrides must reference recognized validation gates.
- **W15 (Production Gate):** Profile `none` requires `metadata.production_ready: false`.

---

## 6. Common Error Codes

| Code | Meaning | Remediation |
|---|---|---|
| `E_SCHEMA_MISSING_FIELD` | A required field is missing. | Check root keys and operation configs. |
| `E_SCHEMA_UNKNOWN_FIELD` | An undeclared field was supplied. | Remove unrecognized attributes. |
| `E_SCHEMA_INVALID_SEMVER` | Version string is not SemVer 2.0.0. | Use `X.Y.Z` (e.g., `1.0.0`, not `1.0`). |
| `E_SCHEMA_INVALID_IDENTIFIER` | An identifier contains invalid characters. | Use only lowercase letters, digits, and hyphens. |
| `E_VALIDATION_LEVEL_INVALID` | Profile `none` used on production recipe. | Set `metadata.production_ready: false`. |
| `E_COMP_FORWARD_DEPENDENCY` | Operation references a later stage. | Reorder operations or correct `depends_on`. |
| `E_COMP_TYPE_MISMATCH` | Incompatible operation state chain. | Ensure operations unify state types. |
| `E_REGISTRY_UNKNOWN_OP` | Operation type is not registered. | Check spelling and installed plugins. |
| `E_REGISTRY_UNKNOWN_VERSION`| Operation version does not exist. | Pin a valid version (e.g. `1.0.0`). |

---

## 7. Python API Integration

The Python API provides direct, programmatic bindings for recipes:

```python
from neuralops.recipe import load_recipe, recipe_hash, save_recipe

# 1. Loading a recipe
recipe = load_recipe("examples/recipes/distilbert-aasvd.yaml")
print(f"Loaded recipe: {recipe.metadata.name} v{recipe.metadata.version}")
print(f"Canonical Hash: {recipe_hash(recipe)}")

# 2. Schema verification
errors = recipe.validate_schema()
if errors:
    raise ValueError(f"Recipe validation failed: {errors}")

# 3. Modifying and saving
recipe.metadata.description = "Updated description for calibrated run."
save_recipe(recipe, "output/distilbert-updated.yaml", format="yaml")
```

---

## 8. CLI Quickstart

NeuralOps exposes recipe management via the `nops` CLI tool:

```bash
# Validate a recipe statically without loading weights:
nops surgery plan --recipe ./examples/recipes/distilbert-aasvd.yaml

# Apply the recipe and produce verifiable execution evidence:
nops surgery apply --recipe ./examples/recipes/distilbert-aasvd.yaml
```
