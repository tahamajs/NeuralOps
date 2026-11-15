# NOTEBOOK AGENT SPECIFICATION: NBA-3
# Model Compression, SVD, Quantization & Pruning (07, 12-14, 33-39)

**Agent ID:** `NBA-3`  
**Role:** Model Compression, SVD, Quantization & Pruning (07, 12-14, 33-39)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-3`  
**Worktree Directory:** `../Project_NBA_3`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/07_master_all_72_compression_engines_quant_and_svd.ipynb`
- `notebooks/12_svd_whitening_spectral_allocation.ipynb`
- `notebooks/13_quantization_pareto_fp8_int4_kmeans.ipynb`
- `notebooks/14_structured_pruning_token_shedding.ipynb`
- `notebooks/33_outlier_aware_smoothquant_pipeline.ipynb`
- `notebooks/34_microscaling_fp4_and_mx_formats.ipynb`
- `notebooks/35_admm_structured_2_4_sparsity.ipynb`
- `notebooks/36_activation_covariance_and_fisher_pruning.ipynb`
- `notebooks/37_cross_layer_matrix_factorization_svd.ipynb`
- `notebooks/38_weight_ternarization_and_1bit_architectures.ipynb`
- `notebooks/39_dynamic_token_shedding_and_early_exit.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Model compression zoo: Whitened SVD, SmoothQuant, FP8/INT4/FP4 quantization, ADMM 2:4 structured sparsity, and Fisher-guided pruning.

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

1. Verify SVD spectral allocation computes real Frobenius error bounds.
2. Ensure SmoothQuant scaling factors are mathematically positive and non-zero.
3. Validate 2:4 sparsity masks strictly enforce 2 zeros per 4 values.
4. Ensure all compression algorithms use `COMPRESSOR_REGISTRY`.

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
git commit -m "fix(notebooks/NBA-3): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-3
Branch: notebooks/NBA-3
Notebooks verified: 11
Tests passing: pytest tests/test_notebooks_execution.py
```
