# Triton kernel layout

`neuralops.core.triton_kernels` is the canonical namespace for NeuralOps
Triton operators. Implementations are grouped by responsibility:

- Files directly in this directory contain shared dispatch, type, registry,
  profiling, and general-purpose attention/state operators.
- `standalone/` contains self-contained fused operators that have their own
  launch configuration and PyTorch reference implementation.
- `neuralops.kernels.triton` is a compatibility facade. New internal imports
  should use `neuralops.core.triton_kernels` or its `standalone` subpackage.

The facade preserves the existing public exports and dotted legacy imports;
it does not own kernel implementations. This keeps kernel code, dispatch
contracts, and reference operators under one architectural root without
forcing unrelated operators into one source file.
