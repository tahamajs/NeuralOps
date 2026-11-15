# NOTEBOOK AGENT SPECIFICATION: NBA-6
# Distributed Parallelism, FSDP2, WSD Schedulers & Optimizers (18, 22-24, 47-50, 66-70, 82)

**Agent ID:** `NBA-6`  
**Role:** Distributed Parallelism, FSDP2, WSD Schedulers & Optimizers (18, 22-24, 47-50, 66-70, 82)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-6`  
**Worktree Directory:** `../Project_NBA_6`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/18_cross_seed_statistical_variance.ipynb`
- `notebooks/22_distributed_tensor_and_context_parallelism.ipynb`
- `notebooks/23_spectral_diagnostics_and_hessian_analysis.ipynb`
- `notebooks/24_advanced_optimizers_muon_stiefel_and_ttt_training.ipynb`
- `notebooks/47_warmup_stable_decay_wsd_scheduler.ipynb`
- `notebooks/48_multi_dataset_ingestion_and_flame_packing.ipynb`
- `notebooks/49_ring_and_zigzag_context_parallelism.ipynb`
- `notebooks/50_activation_checkpointing_strategies.ipynb`
- `notebooks/66_smart_gpu_trainer_and_automax_vram.ipynb`
- `notebooks/67_dora_healing_and_directional_adaptation.ipynb`
- `notebooks/68_ttt_online_test_time_training_adaptation.ipynb`
- `notebooks/69_fisher_micro_steps_and_natural_gradient_damping.ipynb`
- `notebooks/70_preference_alignment_simpo_orpo_and_dynamic_routing_rl.ipynb`
- `notebooks/82_distributed_fsdp2_distillation_and_megatron_hooks.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Distributed scaling: FSDP2, Ring & ZigZag context parallelism, WSD learning rate scheduling, Muon non-Euclidean optimization, and DoRA directional healing.

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

1. Validate simulated TP and CP forward passes in Notebook 22 & 49 without requiring physical multi-node clusters.
2. In Notebook 24: Test Muon Newton-Schulz iterations and verify orthogonality.
3. In Notebook 47: Verify Warmup-Stable-Decay (WSD) scheduler produces three distinct learning rate regimes.
4. In Notebook 67: Verify DoRA directional and magnitude weight decomposition.

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
git commit -m "fix(notebooks/NBA-6): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-6
Branch: notebooks/NBA-6
Notebooks verified: 14
Tests passing: pytest tests/test_notebooks_execution.py
```
