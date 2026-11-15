# NOTEBOOK AGENT SPECIFICATION: NBA-2
# Attention Zoo & Subquadratic Operators (06, 10-11, 26-32)

**Agent ID:** `NBA-2`  
**Role:** Attention Zoo & Subquadratic Operators (06, 10-11, 26-32)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-2`  
**Worktree Directory:** `../Project_NBA_2`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/06_master_all_131_attention_mechanisms_interactive.ipynb`
- `notebooks/10_long_context_extrapolation_32k.ipynb`
- `notebooks/11_cuda_vs_triton_vs_pytorch_matrix.ipynb`
- `notebooks/26_linear_recurrent_associative_scans.ipynb`
- `notebooks/27_kronecker_delta_attention_deep_dive.ipynb`
- `notebooks/28_multi_head_latent_attention_deepseek_mla.ipynb`
- `notebooks/29_state_space_attention_hybrids.ipynb`
- `notebooks/30_differential_attention_noise_cancellation.ipynb`
- `notebooks/31_chunkwise_parallel_attention_kernels.ipynb`
- `notebooks/32_sparse_sinkhorn_and_routing_attentions.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Subquadratic attention operators: GLA, DeltaNet, RetNet, Mamba SSM hybrids, DeepSeek Multi-Head Latent Attention (MLA), and chunkwise associative scans.

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

1. Verify all 131 attention mechanisms in Notebook 06 resolve from `ATTENTION_REGISTRY`.
2. Ensure DeepSeek MLA notebook (28) correctly models compressed KV latent projections (kv_lora_rank, q_lora_rank).
3. Ensure associative scan kernels (26) use numerically stable log-space or chunkwise implementations.
4. Verify long-context extrapolation cells (10) do not OOM on CPU (keep context lengths <= 2048 in test cells).

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
git commit -m "fix(notebooks/NBA-2): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-2
Branch: notebooks/NBA-2
Notebooks verified: 10
Tests passing: pytest tests/test_notebooks_execution.py
```
