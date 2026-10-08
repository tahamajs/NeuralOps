# API Cleanliness Audit

## 1. Ambiguous Registry Names
Audit scan against versioning suffixes/prefixes (`_v\d+`, `v\d+_`, etc.):
- **Attention Modules**: `spectra_v2` (Historical Spectra v2 architectural module)
- **Compressor Modules**: `saes_svd_v2` (Second-generation SAES SVD compression engine)
All legacy exploratory aliases (`aegis_v1` through `v6`, `leka_v1` through `v3`) are successfully deprecated and mapped in `neuralops/core/deprecated_aliases.json`.

## 2. Missing Docstrings
- Total items audited: 2,101 functions/classes/modules without complete docstrings.
- Details compiled into `docs/MISSING_DOCSTRINGS.md`.
- Top files targeted for complete Google-style docstring remediation across `neuralops/core/`, `neuralops/attentions/`, and `neuralops/compression_engine/`.

## 3. Missing Type Hints
- Total items audited: 1,698 function definitions missing argument or return type annotations.
- Details compiled into `docs/MISSING_TYPE_HINTS.md`.
- Public APIs and core orchestrator interfaces prioritized for type safety and `mypy` strictness.

## 4. Recommendations & Actions Taken
1. Re-registered the 14 attention modules to restore the invariant count of 159 attention mechanisms + 73 compressor modules (232 total).
2. Cleaned conflicting compressor alias (`svd_linear`) to guarantee smoke test stability across all 232 modules.
3. Added docstrings and type hints to core public functions and top modules.
