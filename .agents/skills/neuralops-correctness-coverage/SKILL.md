---
name: neuralops-correctness-coverage
description: Select and run correctness coverage for NeuralOps kernels and modules, including gradients, dispatch boundaries, and Triton addressing changes.
---

# NeuralOps Correctness Coverage

Use [neuralops-design-coverage](../neuralops-design-coverage/SKILL.md) when the change needs a new numerical or routing contract. This skill turns that contract into tests; [CONTRIBUTING.md](../../../CONTRIBUTING.md#testing) defines references, tolerances, test structure, and platform conventions.

## Select coverage

Read the affected tests and callers before adding cases. Follow [test organization and naming](../../../CONTRIBUTING.md#test-organization-and-naming): try the existing parameter matrix first, and add a function or file only when its setup or scope requires one. Cover reachable gaps across the applicable dimensions together, rather than testing each neuralopsg only in isolation:

| Dimension   | Cases that can expose different behavior                                         |
| ----------- | -------------------------------------------------------------------------------- |
| Layout      | Dense and varlen, partial chunks, long sequences, asymmetric QK/value dimensions |
| Features    | Gate bounds, raw/post-sigmoid beta, QK normalization, grouped value attention    |
| State       | Initial state, final state, and state gradients where supported                  |
| Numerics    | Supported dtypes, affected conversion/reduction paths, numerical boundaries      |
| Routing     | Default and selected backends, verifier acceptance, rejection, and fallback      |

Compare outputs and final states against the existing reference. Training APIs also need every supported gradient; forward-only APIs must state that backward is unsupported. Use `torch.autograd.gradcheck` where the implementation supports its required precision. Routing tests verify selection and argument forwarding; they do not replace numerical comparisons on the actual backend.

For addressing changes, inspect casts before multiplication and test shapes that expose large offsets, partial tiles, and varlen boundaries. Follow the `tl.int64` and platform requirements in [Triton Kernels](../../../CONTRIBUTING.md#triton-kernels). Exercise changed grid/address paths on affected supported platforms. Distinguish unsupported cases from tests skipped for unavailable hardware or dependencies; record the latter as validation gaps.

The NaN allocation guard covers eligible NeuralOps allocations in operator/module tests, not layer/model/CP tests or every allocation. Add explicit finite-output and finite-gradient checks to adversarial cases, and investigate a poisoned-run failure even when an ad hoc run passes.

## Run the affected paths

Discover dependent tests from the repository root:

```bash
python scripts/find_dependent_tests.py neuralops/ops/kda/chunk.py
NeuralOps_CI_ENV=0 pytest tests/ops/test_kda.py -v
```

The helper lists files; it does not run them. Include reported callers and relevant `tests/modules/`, `tests/layers/`, `tests/models/`, or `tests/context_parallel/` coverage when the change reaches those paths. Inspect skips and warning-only comparisons before calling the run a pass. Record commands, backend neuralopsgs, environment, and results; reproduce failures on the unchanged baseline before labeling them pre-existing.

Read operator math only when needed: [Delta Rule](../../../neuralops/ops/delta_rule/README.md), [Generalized Delta Rule](../../../neuralops/ops/generalized_delta_rule/README.md), [Simple GLA](../../../neuralops/ops/simple_gla/README.md), or [context parallelism](../../../neuralops/ops/cp/README.md).
