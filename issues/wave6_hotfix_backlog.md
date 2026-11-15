# Wave 6 Hotfix Backlog

This backlog tracks findings, edge cases, minor doc mismatches, and improvements discovered during Wave 5 integration, verification, and acceptance testing.
Per Wave 5 protocol, findings are logged here rather than patched inline to prevent regressions.

| Finding ID | Discovered By | Subsystem | Severity | File / Symbol | Description | Recommended Fix |
|---|---|---|---|---|---|---|
| W5-8-01 | W5-8 | attention/linear | Low | neuralops/layers/linear/galerkin_attention.py | GalerkinAttention forward pass raises NotImplementedError | Implement numerical quadrature/Galerkin projection PyTorch CPU/CUDA reference pass |
