# Release Checklist for NeuralOps v1.2.0

## Pre-Release Verification
- [x] All tests pass: `pytest neuralops/tests/ -m "not slow and not gpu"`
- [x] Registry invariant preserved: **159 Attention modules + 73 Compressor modules = 232 Total**
- [x] Zero pyflakes warnings (`F401`, `F841`, `F811`, `F821`)
- [x] All 12 packages import cleanly (`neuralops.core`, `attentions`, `compression_engine`, `benchmarks`, etc.)
- [x] CLI functions verified: `python -m neuralops.benchmarks list/verify`
- [x] Documentation complete: README, FAQ, ARCHITECTURE, PLATFORM_SUPPORT, REPRODUCIBILITY, FAMILIES
- [x] CHANGELOG updated for v1.2.0
- [x] Version consistent across `pyproject.toml`, `neuralops/__init__.py`, `CITATION.cff` (1.2.0)
- [x] Git committed and clean working tree

## Build & Packaging
- [x] Built with `python -m build --no-isolation`
- [x] Twine check passes: `twine check dist/*` (PASSED for both wheel and sdist)
- [x] Wheel size < 5 MB (Actual: **1.9 MB**)
- [x] No tests, raw data, or benchmarks leaked inside wheel package (520 clean package files)

## Production Test Installation
- [x] Clean environment installation verified
- [x] Dynamic module resolution verified (159 + 73 registered modules)
- [x] CLI entry point operational

## PyPI & Distribution Deployment
- [ ] Push tags to origin: `git push origin main --tags`
- [ ] Upload to PyPI / TestPyPI: `twine upload dist/*`
- [ ] Verify live install: `pip install neuralops==1.2.0`
- [ ] GitHub Release created with release notes and wheel artifacts
