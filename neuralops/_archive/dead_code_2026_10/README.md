# Dead Code Archive (October 2026)

## Status
All 25 candidates across `neuralops/` were rigorously audited against:
1. Module Registries (`ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY` -> 159 + 73 keys)
2. Test Suites (`neuralops/tests/` -> active unit & integration test coverage)
3. Documentation and Benchmarking Scripts (`scripts/`, `docs/`)
4. Package build definitions (`pyproject.toml`)

## Findings
Every candidate file is actively referenced, tested, or utilized in benchmarking/analysis workflows:
- Triton kernels (`paged_attention.py`, `flash_decoding.py`, `fused_rope.py`, `block_sparse.py`) are tested in `neuralops/tests/test_triton_kernels.py`.
- Quantization algorithms (`gptq.py`, `awq.py`, `smoothquant.py`, `wanda.py`, `dora_qlora.py`) are tested in compression unit tests.
- Infrastructure modules (`cached_tensors.py`, `suite_runner.py`, `cuda_graph.py`, `fused_norm_attention.py`, `quantized_attention.py`) are integral components.
- Zero orphaned or truly dead modules remain to be archived.
