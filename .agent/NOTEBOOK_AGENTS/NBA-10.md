# NOTEBOOK AGENT SPECIFICATION: NBA-10
# Master Notebook Catalog & Documentation Alignment

**Agent ID:** `NBA-10`  
**Role:** Master Notebook Catalog & Documentation Alignment  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-10`  
**Worktree Directory:** `../Project_NBA_10`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- Master test harness and tooling across all 94 notebooks

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Notebook cataloging and cross-linking: `notebooks/README.md` and `NOTEBOOKS_COMPREHENSIVE_CATALOG_AND_60_EXPANSION_ROADMAP.md`.

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

1. Verify all 94 notebooks have accurate summaries, execution tiers, and topic tags.
2. Confirm links between examples (`examples/`) and notebooks (`notebooks/`).
3. Compile final notebook remediation audit report in `audits/notebooks_remediation_master.md`.

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
git commit -m "fix(notebooks/NBA-10): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-10
Branch: notebooks/NBA-10
Notebooks verified: 0
Tests passing: pytest tests/test_notebooks_execution.py
```
