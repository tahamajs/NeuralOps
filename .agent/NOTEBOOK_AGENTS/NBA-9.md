# NOTEBOOK AGENT SPECIFICATION: NBA-9
# Notebook Test Automation & Headless CI Validator

**Agent ID:** `NBA-9`  
**Role:** Notebook Test Automation & Headless CI Validator  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-9`  
**Worktree Directory:** `../Project_NBA_9`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- Master test harness and tooling across all 94 notebooks

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Test harnesses for Jupyter notebooks: `tests/test_notebook_catalog.py` and `tests/test_notebooks_execution.py`.

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

1. Run and verify `tests/test_notebook_catalog.py`: ensures all 94 notebooks are indexed and `notebooks/extended` symlink is valid.
2. Run and verify `tests/test_notebooks_execution.py`: runs Level (a) AST smoke and structure checks on all 94 notebooks.
3. Generate machine-readable report in `audits/notebooks/ast_validation_results.json`.

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
git commit -m "fix(notebooks/NBA-9): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-9
Branch: notebooks/NBA-9
Notebooks verified: 0
Tests passing: pytest tests/test_notebooks_execution.py
```
