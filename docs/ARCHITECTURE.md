# NeuralOps System Architecture (v1.0.0)

This document formalizes the architectural layer model and strict boundaries governing the NeuralOps framework.

---

## Layer Boundary Model

The framework enforces strict directional dependencies to eliminate circular imports and preserve testability. Lower layers never import from upper layers.

```
       ┌────────────────────────────────────────────────────────┐
       │                Application & CLI Layer                 │
       │       (neuralops.benchmarks.cli, scripts/)            │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │               Runners & Orchestration                  │
       │       (neuralops.runners, run_autonomous_pipeline)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │              Evaluation & Analysis Layer               │
       │       (neuralops.evaluation, neuralops.analysis)       │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │              Model & Operations Domain                 │
       │       (attentions, compression_engine, trainer)        │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │                 Data & Config Layer                    │
       │       (neuralops.data, neuralops.config)               │
       └───────────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────▼────────────────────────────┐
       │                  Core Infrastructure                   │
       │    (neuralops.core, registry, triton_kernels, utils)   │
       └────────────────────────────────────────────────────────┘
```

---

## Module Boundary Rules

1. **Core Infrastructure Boundary**:
   - `neuralops.core` must never import from `attentions`, `compression_engine`, `trainer`, `evaluation`, or `runners`.
   - Shared Triton types reside in `neuralops.core.triton_kernels._types.py` without requiring Triton GPU runtime imports.

2. **Domain Isolation**:
   - `neuralops.attentions` and `neuralops.compression_engine` are mutually isolated. They communicate only through `neuralops.core.registry`.

3. **Lazy Import Pattern**:
   - Registry instances lazily resolve target modules on instantiation to maintain cold start import latency under 3 seconds.
