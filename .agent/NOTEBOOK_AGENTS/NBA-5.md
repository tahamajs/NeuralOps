# NOTEBOOK AGENT SPECIFICATION: NBA-5
# Hugging Face Export, SafeTensors, GGUF & Hub Upload (21, 65, 87)

**Agent ID:** `NBA-5`  
**Role:** Hugging Face Export, SafeTensors, GGUF & Hub Upload (21, 65, 87)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-5`  
**Worktree Directory:** `../Project_NBA_5`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/21_model_export_onnx_safetensors_surgery_manifest.ipynb`
- `notebooks/65_huggingface_polymorphic_generation_cache_bridge.ipynb`
- `notebooks/87_universal_model_export_safetensors_and_gguf_container.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Multi-format model export: SafeTensors, ONNX, GGUF, Hugging Face Hub uploader, and polymorphic generation cache bridge.

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

1. Verify SafeTensors export and reload parity in Notebook 21 & 87.
2. Connect to `neuralops.hub.uploader.AsyncHFHubUploader` with safe `HF_TOKEN` credential checks.
3. In Notebook 65: Verify `neuralops.surgery.hf_cache_bridge` cleanly bridges HF DynamicCache with subquadratic recurrent states.
4. Verify GGUF export cells skip gracefully if `gguf` package is not installed.

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
git commit -m "fix(notebooks/NBA-5): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-5
Branch: notebooks/NBA-5
Notebooks verified: 3
Tests passing: pytest tests/test_notebooks_execution.py
```
