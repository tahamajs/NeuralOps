# Cross-Domain Bug Findings — Subagent W4-C1

## 2026-11-15T02:26:00Z
File: neuralops/compression/svd/hnc_svd_continuation.py
Line: 462
Symptom: Hardcoded `"cuda:0"` device string fallback (`self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))`) violates Rule 2.5 and Rule 10.
Suspected bug ID: P1-C1-CROSS-1
Suggested fix: Replace `"cuda:0"` with `"cuda"` or dynamic device resolution helper without hardcoded device ordinal 0.

## 2026-11-15T02:35:00Z
File: neuralops/compression/low_rank/cur_compressor.py
Line: 257
Symptom: `is_linear = isinstance(module, nn.Linear) or "linear" in cls_name` selects container classes named `*Linear*` (e.g. `LinearWrapper`, `LinearContainer`) without checking `hasattr(target_mod, "weight") and target_mod.weight is not None`, causing `AttributeError: '<Container>' object has no attribute 'weight'` during compression.
Suspected bug ID: P0-C1-CROSS-2
Suggested fix: Add `target_mod = getattr(module, "base_layer", getattr(module, "linear", module))` and `if not hasattr(target_mod, "weight") or target_mod.weight is None: return False` to `_is_target_module`.

## 2026-11-15T02:37:00Z
File: neuralops/compression/quantization/smoothquant.py
Line: 169
Symptom: `module.forward = _smoothed_forward` dynamically binds a local closure capturing `module` on `nn.Linear` instances, which causes `copy.deepcopy` to fail when tensors are non-leaf nodes (`Only Tensors created explicitly by the user support deepcopy`).
Suspected bug ID: P1-C1-CROSS-3
Suggested fix: Use `SmoothQuantLinear` module replacement instead of closure patching on `module.forward`.
