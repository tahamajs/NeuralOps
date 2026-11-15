# Wave 4 Agent W4-L3: Cross-Domain Bugs Log

## 2026-11-15T02:35:00Z
File: neuralops/layers/dense/full_attention.py
Line: 368
Symptom: `FullAttention.forward()` accepts `past_key_value: Any | None = None` in its argument signature, but the tensor is completely ignored in the body of `forward()`. It is never read (`_read_cache` is missing), never updated, and never returned, breaking autoregressive generation when KV caching is enabled.
Suspected bug ID: P0-L3-CROSS-1
Suggested fix: Implement `_read_cache` and `_write_cache` logic (mirroring `GroupedQueryAttention` in `neuralops/layers/modern/gqa_attention.py`) to properly concatenate past KV cache and return updated cache state.

## 2026-11-15T02:35:00Z
File: neuralops/core/_compat_importer.py
Line: 314
Symptom: `NeuralOpsCompatibilityRedirectFinder.find_spec` contains logic in `_resolve_canonical` to resolve flat subpackage lookups like `neuralops.layers.<module>` to `neuralops.layers.<subpkg>.<module>`, but `find_spec` itself only triggers for legacy prefixes (`neuralops.attentions.` and `neuralops.compression_engine.`). Flat lookups like `import neuralops.layers.full_attention` do not get redirected and raise `ModuleNotFoundError`.
Suspected bug ID: P1-L3-CROSS-2
Suggested fix: In `find_spec`, check if `fullname.startswith("neuralops.layers.")` or `fullname.startswith("neuralops.compression.")` and delegate to `_resolve_canonical`.
