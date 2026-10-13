## Pull Request Description

### Summary of Changes
<!-- Provide a brief, high-level summary of the changes introduced by this PR. -->

### Category
- [ ] Attention Mechanism (`neuralops/attentions/`)
- [ ] Compression Engine (`neuralops/compression_engine/`)
- [ ] Triton Kernel / Op Registry (`neuralops/core/triton_kernels/`)
- [ ] Production Export & Serving (`neuralops/export/`)
- [ ] MLOps & Benchmark CLI (`neuralops/benchmarks/`)
- [ ] Bug Fix / Documentation

---

### Verification & Quality Checklist
- [ ] Verified that `python -m neuralops.benchmarks verify` passes.
- [ ] Ran test suite: `pytest tests/ -v`.
- [ ] Preserved 100% backward compatibility for canonical registry keys and legacy aliases.
- [ ] If a Triton kernel was added/modified, verified that a mathematically equivalent PyTorch CPU/MPS fallback is provided.
- [ ] If custom ops were added, verified FakeTensor / meta shape registrations for `torch.compile`.
- [ ] Added or updated unit tests and docstrings.
