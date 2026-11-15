# NOTEBOOK AGENT SPECIFICATION: NBA-8
# Evaluation Harness, NIAH, Diagnostics & Production Pipelines (08, 15, 19, 25, 51-53, 57-62, 64, 75-81, 83-84, 88-90)

**Agent ID:** `NBA-8`  
**Role:** Evaluation Harness, NIAH, Diagnostics & Production Pipelines (08, 15, 19, 25, 51-53, 57-62, 64, 75-81, 83-84, 88-90)  
**Scope:** Notebooks Remediation & Validation  
**Target Branch:** `notebooks/NBA-8`  
**Worktree Directory:** `../Project_NBA_8`  
**Progress Tracker:** `.agent/NOTEBOOK_PROGRESS.json`  

---

## 1. OWNED NOTEBOOKS & SCOPE OF RESPONSIBILITY

You are responsible for auditing, remediating, and validating the following Jupyter notebooks:
- `notebooks/08_six_pillar_distillation_and_surgical_recovery.ipynb`
- `notebooks/15_downstream_glue_lra_evaluation.ipynb`
- `notebooks/19_publication_figures_latex_generator.ipynb`
- `notebooks/25_long_context_niah_cka_similarity_and_eval_harness.ipynb`
- `notebooks/51_cross_model_representation_alignment.ipynb`
- `notebooks/52_mixture_of_attentions_routing.ipynb`
- `notebooks/53_knowledge_distillation_ablation_matrix.ipynb`
- `notebooks/57_numerical_precision_and_nan_inf_stress_tests.ipynb`
- `notebooks/58_synthetic_stress_testing_and_chaos_engineering.ipynb`
- `notebooks/59_reproducible_artifact_generation_and_latex_sync.ipynb`
- `notebooks/60_end_to_end_industrial_llm_transformation.ipynb`
- `notebooks/61_universal_surgery_director_and_ten_laws.ipynb`
- `notebooks/62_deepseek_mla_dual_stream_latent_surgery.ipynb`
- `notebooks/64_zero_oom_streaming_distillation_and_shadow_twin.ipynb`
- `notebooks/75_long_horizon_extrapolation_32k_and_kv_complexity.ipynb`
- `notebooks/76_pipeline_transform_master_planner_and_transactional_rollback.ipynb`
- `notebooks/77_numerical_guardrails_static_validation_and_safety_gates.ipynb`
- `notebooks/78_experiment_logging_telemetry_tracker_and_results_saver.ipynb`
- `notebooks/79_comprehensive_evaluation_harness_niah_passkey_and_ab_testing.ipynb`
- `notebooks/80_neuralops_cli_ecosystem_and_automated_evidence_reporting.ipynb`
- `notebooks/81_autonomous_circuit_breaker_fsm_and_runtime_resilience.ipynb`
- `notebooks/83_fla_flash_linear_attention_bidirectional_bridge.ipynb`
- `notebooks/84_canonical_recipe_declarative_schema_and_versioning.ipynb`
- `notebooks/88_hybrid_attention_architectures_and_multi_branch_fusion.ipynb`
- `notebooks/89_flame_multi_dataset_ingestion_and_zero_pad_varlen_packing.ipynb`
- `notebooks/90_layerwise_spectral_decay_and_pareto_sweeper_analysis.ipynb`

---

## 2. DOMAIN & ARCHITECTURAL CONTEXT

Comprehensive evaluations: GLUE/LRA benchmarks, CKA representational similarity, Needle In A Haystack (NIAH), circuit breakers, and end-to-end recipe transforms.

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

1. In Notebook 25 & 79: Ensure NIAH passkey evaluation uses real language tokens and causal masking (no fake randn).
2. In Notebook 81: Verify circuit breaker FSM transitions across NORMAL, THROTTLED, and TRIPPED states.
3. In Notebook 84: Validate declarative recipe schema parsing against Pydantic models.
4. Ensure all cells complete without unbounded loops or memory leaks.

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
git commit -m "fix(notebooks/NBA-8): remediate and validate <assigned_notebooks>

- Validated AST and JSON structure
- Fixed broken imports and fake data patterns
- Verified CPU fallback execution"
```

Completion report:
```
COMPLETE: NBA-8
Branch: notebooks/NBA-8
Notebooks verified: 26
Tests passing: pytest tests/test_notebooks_execution.py
```
