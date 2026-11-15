# NOTEBOOK AGENT SPECIFICATION: NBA-1
# Core System Walkthroughs & Architecture Foundations (00-05)

**Agent ID:** `NBA-1`  
**Role:** Core System Walkthroughs & Architecture Foundations (00-05)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-1`  
**Worktree Directory:** `../Project_NBA_1`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/00_master_neuralops_complete_system_walkthrough.ipynb`
- `notebooks/01_attention_surgery_and_lora.ipynb`
- `notebooks/01_quickstart_surgery_and_healing.ipynb`
- `notebooks/02_roofline_and_triton_kernels.ipynb`
- `notebooks/02_unified_training.ipynb`
- `notebooks/03_custom_plugins_and_extensibility.ipynb`
- `notebooks/04_end_to_end_3level_benchmarking.ipynb`
- `notebooks/05_master_108_cuda_and_26_triton_kernels_benchmark.ipynb`
- `notebooks/05_paper_empirical_results_master.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Core NeuralOps architecture, unified training loop, 3-level kernel dispatching, and empirical paper benchmark reproduction.

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

1. Validate notebook JSON syntax and ensure all code cells pass Python AST parsing.
2. Verify that attention surgery cells use `neuralops.replace_attention` without deprecated kwargs.
3. Check LoRA adaptation cells against `neuralops.models.adapters.lora.LoRALinear`.
4. Ensure plots and figures render correctly or execute CPU smoke paths when CUDA is absent.
5. Fix broken imports from legacy paths.

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
git commit -m "fix(notebooks/NBA-1): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-1
Branch: notebooks/NBA-1
Notebooks verified: 9
Tests passing: pytest tests/test_notebooks_execution.py
```
