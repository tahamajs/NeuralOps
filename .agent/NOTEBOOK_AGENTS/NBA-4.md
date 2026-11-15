# NOTEBOOK AGENT SPECIFICATION: NBA-4
# Production Serving: vLLM, SGLang, TRT-LLM & Speculative Fusion (09, 17, 20, 44, 54-56, 85-86)

**Agent ID:** `NBA-4`  
**Role:** Production Serving: vLLM, SGLang, TRT-LLM & Speculative Fusion (09, 17, 20, 44, 54-56, 85-86)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-4`  
**Worktree Directory:** `../Project_NBA_4`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/09_production_serving_paged_cache_and_speculative_decoding.ipynb`
- `notebooks/17_speculative_paged_kv_throughput.ipynb`
- `notebooks/20_production_serving_vllm_sglang_trtllm_integrations.ipynb`
- `notebooks/44_tensorrt_engine_building_and_plugins.ipynb`
- `notebooks/54_kv_cache_compression_and_eviction_policies.ipynb`
- `notebooks/55_speculative_decoding_target_draft_synergy.ipynb`
- `notebooks/56_vllm_sglang_custom_op_integration.ipynb`
- `notebooks/85_speculative_decoding_engine_and_paged_state_fusion.ipynb`
- `notebooks/86_cuda_graph_decode_runner_and_zero_overhead_inference.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

High-throughput serving engines: vLLM PagedAttention custom backends, SGLang RadixAttention, TensorRT-LLM plugin generation, and speculative decoding.

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

1. In Notebook 20 & 56: Connect to `neuralops.integrations.vllm` and `neuralops.integrations.vllm_backend`.
2. Ensure cells gracefully report `[SKIP: vLLM not installed]` on CPU without crashing.
3. Validate PagedStateTable allocation and block indexing logic in Notebook 09 and 85.
4. Verify TensorRT-LLM C++ plugin synthesis compiles or generates clean C++ code in Notebook 44.

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
git commit -m "fix(notebooks/NBA-4): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-4
Branch: notebooks/NBA-4
Notebooks verified: 9
Tests passing: pytest tests/test_notebooks_execution.py
```
