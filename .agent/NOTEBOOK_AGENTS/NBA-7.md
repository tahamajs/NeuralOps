# NOTEBOOK AGENT SPECIFICATION: NBA-7
# Hardware Kernels, Triton JIT, Roofline Analysis & Multi-Arch (16, 40-43, 45-46, 63, 71-74)

**Agent ID:** `NBA-7`  
**Role:** Hardware Kernels, Triton JIT, Roofline Analysis & Multi-Arch (16, 40-43, 45-46, 63, 71-74)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-7`  
**Worktree Directory:** `../Project_NBA_7`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/16_roofline_ada_rtx4070_saturation.ipynb`
- `notebooks/40_cuda_cpp_warp_shuffle_and_shared_memory.ipynb`
- `notebooks/41_triton_2d_tiling_without_block_pointers.ipynb`
- `notebooks/42_huawei_ascend_npu_cann_adaptation.ipynb`
- `notebooks/43_amd_rocm_hip_cross_compilation.ipynb`
- `notebooks/45_register_pressure_and_occupancy_tuning.ipynb`
- `notebooks/46_gluon_experimental_dialect_kernels.ipynb`
- `notebooks/63_fx_graph_rewriter_and_inductor_compiler.ipynb`
- `notebooks/71_triton_26_hardware_kernels_comprehensive_benchmark.ipynb`
- `notebooks/72_llm_memory_bandwidth_and_l2_cache_saturation.ipynb`
- `notebooks/73_multiprecision_numerical_validation_fp8_bf16_fp32.ipynb`
- `notebooks/74_dynamic_range_stress_testing_and_nan_inf_resilience.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Hardware kernel optimization: Triton 2D block tiling, CUDA C++ warp shuffles, roofline arithmetic intensity bounds, and TorchInductor FX graph compilation.

---

## 3. STRICT OPERATIONAL RULES FOR NOTEBOOKS

1. **Syntactic & JSON Integrity:**
   - Every `.ipynb` file MUST be valid JSON conforming to Jupyter Notebook Format v4.
   - Every code cell MUST parse cleanly with `ast.parse()` without SyntaxError.

2. **No Fake / Mock Data for Empirical Claims:**
   - Notebooks claiming to evaluate models, perplexity, throughput, or memory MUST use real models (`TinyClassifier`, HF AutoModel) and real tokens.
   - `torch.randn` is forbidden for evaluation inputs or attention matrices.

3. **Graceful Hardware Degradation:**
   - Cells requiring GPU/Triton MUST check `torch.cuda.is_available()`.
   - If CUDA is absent, print `[SKIP: CUDA Required]` and provide a CPU reference execution path. Never crash on CPU.

4. **vLLM & Hugging Face Integration Integrity:**
   - When importing vLLM (`neuralops.integrations.vllm` / `vllm_backend`), check if `vllm` is installed. If not, inspect contracts statically and report SKIP.
   - When exporting to Hugging Face Hub, check `HF_TOKEN` safely without leaking secrets.

---

## 4. GRANULAR AUDIT & REMEDIATION TASKS

1. Verify all Triton kernels have CPU fallbacks when CUDA is unavailable.
2. Check roofline analysis calculations (FLOPs / Byte) against known hardware specs (RTX 4070, RTX 3090, A100).
3. In Notebook 63: Verify FX graph rewriter replaces standard SDPA with linear attention subgraphs.
4. In Notebook 74: Ensure NaN/Inf guardrails catch gradient overflows.

---

## 5. VERIFICATION COMMANDS

```bash
# 1. Validate notebook JSON and AST structure:
pytest tests/test_notebooks_execution.py -q -k "structure"

# 2. Validate notebook catalog:
pytest tests/test_notebook_catalog.py -q

# 3. CPU smoke execution:
pytest tests/test_notebooks_execution.py -q -k "smoke"
```

---

## 6. COMMIT & COMPLETION REPORT

Commit format:
```bash
git add notebooks/<assigned_notebooks>
git commit -m "fix(notebooks/NBA-7): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-7
Branch: notebooks/NBA-7
Notebooks verified: 12
Tests passing: pytest tests/test_notebooks_execution.py
```
