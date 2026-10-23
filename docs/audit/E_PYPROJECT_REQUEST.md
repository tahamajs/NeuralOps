# pyproject.toml Modification Request from Agent E (Integration Honesty)

**Target**: Agent D (Packaging & Dependencies Specialist) / Multi-Agent Orchestrator  
**Audit ID**: P006  
**Date**: 2026-10-23  
**Status**: Pending Agent D execution  

---

## 1. Summary of Request

Remove the `[project.entry-points."vllm.general_plugins"]` section from `pyproject.toml`.

## 2. Exact Lines to Remove in `pyproject.toml`

```toml
[project.entry-points."vllm.general_plugins"]
neuralops_vllm = "neuralops.integrations.vllm:register_neuralops_vllm_plugin"
```

(Lines 139-140 in `pyproject.toml`).

## 3. Engineering & Forensic Rationale

1. **Honest Integration Contract (P006)**:
   In NeuralOps v1.6.0, native vLLM integration has been audited as an experimental scaffold whose full GPU backend lifecycle, paged-cache writes, and checkpoint loading are deferred to NeuralOps v2.0 (see `docs/roadmap/VLLM_PLAN.md`).
2. **Prevent Accidental Plugin Invocation**:
   Declaring `vllm.general_plugins` in `pyproject.toml` causes standard vLLM installations to auto-discover and load `register_neuralops_vllm_plugin()` at startup, emitting a `DeprecationWarning` and returning `False`. Removing the entry point prevents uninvited plugin registration attempts in upstream vLLM inference workers while keeping the module available for roadmap exploration.
3. **Ownership Boundary Compliance**:
   Per `ORCHESTRATION_OWNERSHIP.md`, Agent D strictly owns `pyproject.toml` to prevent merge conflicts across concurrent Wave 2 agents. Agent E therefore requests this modification via this formal document.
