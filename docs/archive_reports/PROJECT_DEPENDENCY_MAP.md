# NeuralOps Dependency Map

This map records the verified execution relationships used to navigate the repository. It intentionally distinguishes static references from runtime behavior.

## Primary execution path

```text
CLI / script / test
  -> runner or benchmark adapter
    -> public NeuralOps API
      -> registry lookup in neuralops/core/registry.py
        -> attention/compressor implementation
          -> backend dispatch
            -> CUDA / Triton / PyTorch reference tier
```

## Registry relationships

- `ATTENTION_REGISTRY` and `COMPRESSOR_REGISTRY` are defined in `neuralops/core/registry.py`.
- Registration decorators and registry accessors are also defined there.
- Canonical implementations are distributed across family directories under `neuralops/layers/` and `neuralops/compression/`.
- `neuralops/attentions/` and `neuralops/compression_engine/` must be treated as compatibility or alternate API surfaces until each symbol's caller graph proves otherwise.
- `neuralops/benchmarks/static_catalog.json` is a static catalog, not a replacement for runtime registry resolution.

## Backend relationships

```text
High-level operator
  -> neuralops/ops/cuda/_fallback.py
       -> native CUDA implementation
       -> Triton implementation
       -> PyTorch reference implementation
```

Any fallback must remain observable and numerical parity must be checked before a result is reported as valid.

## Benchmark relationships

- `scripts/benchmarks/` contains standalone executable benchmark workflows.
- `neuralops/benchmarks/` contains reusable importable benchmark infrastructure.
- `tests/benchmarks/` validates benchmark contracts and integrity behavior.
- `tests/gpu_exhaustive/` exercises the remote GPU matrix and must run through PBS on `N3090`.

## Cleanup decision rule

No module is a deletion candidate merely because it is old, duplicated by name, or outside the preferred directory. Before proposing removal, verify:

1. no Python import or package export references it;
2. no registry, CLI, benchmark, notebook, config, or test references it;
3. no compatibility promise requires it;
4. its behavior is represented by a canonical replacement;
5. documentation and changelog references can be updated without breaking history.

The caller graph and cleanup candidate list remain separate deliverables so that evidence is preserved before any destructive change.
