# NeuralOps inference-engine integration audit
Date: 2026-10-21
Local commit: `215f8f893070317a9926af4925fcf24b337a0e11`
Remote target: `taha@192.168.1.100:/mnt/d/SVD_linear_Attention`

## 1. Executive summary
**Verdict:** The repository contains integration/export scaffolding, but this audit cannot certify that surgically modified models run end-to-end in the requested production engines. The target GPU host refused SSH, so no package versions, runtime loading, generation, fullgraph/dynamic compile runs, or per-attention tests were obtained. Source evidence supports partial status for HF, `torch.compile`, vLLM, SGLang, and ONNX; TensorRT-LLM remains a scaffold; the inspected GGUF path does not create GGUF.

Top blockers: (1) no remote runtime evidence; (2) standard HF AutoModel reconstruction after third-party model surgery is not demonstrated; (3) vLLM/SGLang adapters are not proven against a real server and surgery checkpoint; (4) TensorRT no-SDK path emits manifest/marker rather than a verified engine; (5) GGUF option writes metadata plus safetensors/bin, not GGUF.

Planning-only effort estimates to minimal support for ~20 variants: HF 2–5 engineer-days; torch.compile 3–8 engineer-days; vLLM 3–6 engineer-weeks; SGLang 2–5 engineer-weeks; TensorRT-LLM 4–8 engineer-weeks; llama.cpp 3–8 engineer-weeks; ONNX 1–3 engineer-weeks. These are engineering estimates, not measured commitments.

## 2. Repository inventory
- `neuralops/integrations/`: present (HF AutoModel, compile, serving, SGLang cache helper, TRT-LLM, TGI, vLLM).
- `neuralops/core/export/`: absent; exporter is `neuralops/export/`.
- Standalone packages `neuralops-vllm/`, `neuralops-sglang/`, `neuralops-tensorrt-llm/`, `neuralops-llamacpp/`: absent.
- `torch.library`: 26 custom-op decorators and 30 `register_fake` occurrences in `neuralops/core/triton_kernels/op_registry.py`. Repository-wide grep-like search gave 60 matching lines across two files, not 60 registered ops.
- Exact class `AttentionSpec`: absent. Related classes: `ArchitectureSpec`, `AttentionTargetSpec`, `HybridAttentionSpec`; no per-engine 131-variant metadata contract was found.
- Three requested `docs/INTEGRATION_*.md` files: absent.
- README catalog has 131 numbered attention rows. Runtime Python registry length was not queried: local Python execution is prohibited by repository instructions and remote SSH failed.
- Remote host: `ssh: connect to host 192.168.1.100 port 22: Connection refused` (exit 255). Exact structural output is in `raw_logs/`.

## 3. HF Transformers — PARTIAL
`neuralops/integrations/hf_automodel.py` provides patch/AutoModel registration. `neuralops/export/exporter.py` stores NeuralOps config/manifest, weights and attempts AutoMap wiring; `load_exported_model()` replays surgery. Test definitions cover custom NeuralOps save/load for six variants and GPT-2+`cosformer` through the NeuralOps exporter, but were not run. They do not prove ordinary `AutoModelForCausalLM.from_pretrained()` reconstructs arbitrary surgically modified GPT-2 or the requested 5 variants. Exact `config.json` diff/errors: not captured.
Variant results (`cosformer`, `linear_attention`, `gla_attention`, `mamba_block`, `alibi_attention`): all ⏳ NOT RUN.
Metadata required for reconstruction includes base architecture/config, per-layer attention id and kwargs, surgery placement, compression/adapters, and import/AutoMap code. A manifest schema exists, but external-loader compatibility is unverified.

## 4. torch.compile — PARTIAL / UNTESTED
`compile_attention_module()` wraps `torch.compile(module, mode=..., dynamic=...)` and returns the original module if invocation raises. There are 26 custom-op definitions and fake registration references, and `tests/test_zero_graph_break_dynamo.py` covers selected custom ops, not full modified GPT-2 variants. For each of the five variants, fullgraph=True, fullgraph=False and dynamic=True are ⏳ NOT RUN; therefore no error message or root-cause claim is supplied.

## 5. vLLM — PARTIAL / UNTESTED
`neuralops/integrations/vllm.py` contains backend metadata/interface, paged-state operations, a model wrapper, and a plugin registration function; `pyproject.toml` declares a vLLM general-plugin entry point. The wrapper constructs generic `nn.TransformerEncoderLayer` layers rather than reconstructing arbitrary surgically modified HF checkpoints. Tests exercise toy prefill/decode, wrapper output shape, and registration return type; they do not start `vllm.LLM` on the surgery checkpoint. Installed version and exact server error: NOT CHECKED. No per-attention vLLM compatibility is asserted.

## 6. SGLang — PARTIAL / UNTESTED
`neuralops/integrations/sglang.py` implements a NeuralOps radix-prefix/cache helper. No direct SGLang import or standalone plugin package was found in the inspected paths. Tests exercise cache matching/eviction, not SGLang server loading/generation. Package version and runtime error: NOT CHECKED.

## 7. TensorRT-LLM — SCAFFOLD / NOT PRODUCTION
The integration contains plugin descriptors and source-generation scaffolding. Source-defined exact error when TensorRT is unavailable:
```
TensorRT-LLM plugin compilation is planned for NeuralOps v2.0.0.
See docs/roadmap/TENSORRT_LLM_PLAN.md for the implementation roadmap.
Current workarounds: use vLLM or HF serving adapters instead.
```
The no-TensorRT `export_engine()` branch writes a JSON manifest and bytes beginning `NEURALOPS_TRT_ENGINE_V1`; this is not a verified serialized engine. Tests expect `build_plugin()` to raise in the no-SDK path. Installed versions, `auto_deploy`, and runtime: NOT CHECKED.

## 8. llama.cpp / GGUF — NOT IMPLEMENTED IN INSPECTED PATH
No standalone llama.cpp package or `convert_hf_to_gguf` converter was found. The `format="gguf"` branch writes `gguf_metadata.json` and safetensors or `model.bin`, not a GGUF container. No runtime binary or artifact scan could be performed remotely. Architecture-key requirements are omitted rather than guessed.

## 9. ONNX Runtime — PARTIAL
The exporter attempts ONNX export and retries with `dynamo=False`. On failure it writes the text marker `# ONNX export fallback marker: {e}` into `model.onnx`, which is not a valid ONNX protobuf. An in-tree test definition validates ONNX for a tiny directly compressed linear model, not surgically replaced attention variants. No test/version/runtime result was obtained.

## 10. Coverage matrix (131 × 7)
`attention_coverage_matrix.csv` lists the 131 names in the README's numbered catalog, with one column per requested engine. Every per-attention cell is ⏳ UNTESTED because remote runtime tests did not run. This is intentionally conservative: code presence is not end-to-end compatibility. Per-engine status counts: HF 0 ✅/0 🟡/0 ❌/131 ⏳; torch.compile 0/0/0/131; vLLM 0/0/0/131; SGLang 0/0/0/131; TRT-LLM 0/0/0/131; llama.cpp 0/0/0/131; ONNX 0/0/0/131. Static engine verdicts are PARTIAL (HF, compile, vLLM, SGLang, ONNX), SCAFFOLD (TRT-LLM), NOT IMPLEMENTED in inspected export path (llama.cpp).

## 11. Root-cause analysis (source-based hypotheses only)
- **Config/reconstruction:** custom NeuralOps replay loader exists; generic external AutoModel reconstruction of replaced third-party model layers is not demonstrated. Broad effect.
- **Custom-op/graph capture:** ops/fake registrations are selected-operation-level, not per-131 full-model proof.
- **Engine registration:** vLLM plugin is present but real loading untested; SGLang adapter is cache-oriented with no verified server model registration.
- **State/cache:** recurrent states and standard paged KV cache require distinct contracts; no per-attention cache compatibility matrix was found.
- **Backend/format absent or scaffolded:** GGUF serializer absent in inspected branch; TensorRT build path explicitly planned/placeholder without SDK.
- Affected per-attention identities are not assigned to failure buckets without executing each module.

## 12. Effort estimates
See Section 1. Estimates target minimal support for around 20 curated variants (dense/local, linear, recurrent, GQA, sparse/hybrid), assuming a stable engine-version matrix and accessible GPU. They are planning judgments, not test results.

## 13. Priority
1. Make and test the HF checkpoint reconstruction contract.
2. Validate a small `torch.compile` subset in fullgraph and dynamic modes.
3. Complete a narrow causal/cache-compatible vLLM path.
4. Add actual SGLang model/backend registration after defining state ownership.
5. Validate a supported ONNX attention subset and never treat marker files as successful models.
6. Defer TRT-LLM and llama.cpp until real engine/converter paths are available.

## 14. Evidence appendix
Exact local structural output and SSH failure are in `raw_logs/section_a_static.txt` and `raw_logs/remote_gpu_attempt.txt`. Source/test references are listed in the sections above. The README says HF and ONNX are “Production Ready”; this audit cannot confirm those claims. README calls vLLM/SGLang experimental and TRT-LLM roadmap/scaffold; static source agrees with that caution. Test files were inspected, not executed. The initial audit made no code changes; a later local correction is recorded below. Reports remain local and were not committed because repository policy forbids committing ignored `docs/` content.

## Post-audit local correction (2026-10-21)

The ONNX marker writer and metadata-only GGUF export branch were changed locally to fail honestly rather than publish false-format artifacts. These edits have not been executed or tested. The TensorRT sentinel writer and random-output runner remain present because the subsequent mission explicitly prohibited touching `neuralops/integrations/tensorrt_llm.py` without GPU access. The README integration statuses were downgraded. No runtime verdict has changed to verified; see `SILENT_FAILURE_INVENTORY.md` and `../integrations/DEFERRED_ENGINES.md`.

