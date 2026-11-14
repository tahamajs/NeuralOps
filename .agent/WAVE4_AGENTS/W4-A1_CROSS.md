## 2026-11-15T02:35:00Z
File: neuralops/core/registry.py
Line: 350
Symptom: `randomized_svd_compressor` is defined and registered in `neuralops/compression/svd/randomized_compression.py`, but is missing from `_COMPRESSOR_FILE_MAP` in `neuralops/core/registry.py`. Because the module file name (`randomized_compression.py`) differs from the key (`randomized_svd_compressor`), lazy-loading in `COMPRESSOR_REGISTRY.resolve("randomized_svd_compressor")` fails to locate the module unless eager loading is performed.
Suspected bug ID: P1
Suggested fix: Add `"randomized_svd_compressor": "svd.randomized_compression"` to `_COMPRESSOR_FILE_MAP` in `neuralops/core/registry.py`.
