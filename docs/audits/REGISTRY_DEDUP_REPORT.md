# Registry Deduplication Report

## Overview
Successfully enforced the "ONE FILE → ONE CLASS → ONE CANONICAL REGISTRY KEY" rule across the entire `neuralops/attentions/` and `neuralops/compression_engine/` codebase.

## Actions Taken
1. **Deduplication Strategy**: Analyzed 53 files in `ATTENTION_REGISTRY` that contained multiple registry keys for a single class or injector functions. Removed all extraneous `@ATTENTION_REGISTRY.register` and `@COMPRESSOR_REGISTRY.register` decorators.
2. **Backward Compatibility**: Automatically added all removed keys (e.g., legacy codenames, `_inject`, `_attention` suffixes) to `neuralops/core/deprecated_aliases.json` mapped to the canonical file key.
3. **Consolidation**:
   - `decay_linear_attention.py` and `gated_linear_attention.py` were identical classes defining `DecayLinearAttention`. Replaced the former with the updated implementation of the latter and removed `gated_linear_attention.py` entirely, resolving a file conflict.
   - Replaced all imports and instantiations of `gated_linear_attention` across the test suite and evaluation code with `decay_linear_attention` to preserve tests.
4. **Injector Functions**: Stripped `@COMPRESSOR_REGISTRY.register` from `a3_compressor_factory`, `huffman_factory`, and `adaptive_ranks_factory`, ensuring only the primary class holds the registry decorator.
5. **No Broken Syntax**: Validated Python code syntax for all modified modules, assuring zero trailing parentheses or indentation faults.

## Outcome
- Duplicate keys in `ATTENTION_REGISTRY`: **0**
- Duplicate keys in `COMPRESSOR_REGISTRY`: **0**
- `deprecated_aliases.json` updated with backwards-compatible aliases.
