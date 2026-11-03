# `.agents` Scaling Routing Matrix

This is the file-by-file routing index for scaling work. An agent must read the row for every file it plans to change and the linked skill/rule before editing. A file marked **read-only for scale** is evidence or policy; update it only when its documented fact is actually changed.

## Root, hooks, and rules

| File | Scaling responsibility | Change rule |
|---|---|---|
| `.agents/NOTICE` | Attribution and distribution notice | Read; do not alter for workload scaling. |
| `.agents/README.md` | Skill navigation | Update when adding or renaming a scaling workflow. |
| `.agents/hooks/pre-commit.mdc` | Pre-commit enforcement | Read before committing scaled changes; change only hook policy. |
| `.agents/hooks/post-commit.mdc` | Post-commit automation | Read before committing; never use it to hide failed scale verification. |
| `.agents/rules/neuralops-benchmark-sources.mdc` | Benchmark source-of-truth policy | Read for every benchmark scale change; update only if source ownership changes. |
| `.agents/rules/neuralops-canonical-counts.mdc` | 131/72/26 canonical invariants | Read when changing breadth; never alter counts to make a partial run pass. |
| `.agents/rules/neuralops-commit-protocol.mdc` | Commit boundaries and attribution | Read before commit; one logical scaling change per commit. |
| `.agents/rules/neuralops-doc-verification.mdc` | Documentation evidence gates | Read before updating README, manifests, or reports. |
| `.agents/rules/neuralops-kernel-quality.mdc` | Kernel parity and performance quality | Read for CUDA/Triton scale; no fake backend or silent fallback. |
| `.agents/rules/neuralops-no-local-python.mdc` | Local execution restriction | Read before tests; GPU/runtime verification belongs on HPC. |
| `.agents/rules/neuralops-testing-protocol.mdc` | Test selection and evidence | Read before changing test scale or acceptance gates. |
| `.agents/rules/neuralops-thesis-agents-compliance.mdc` | Thesis/agent compliance | Read when scale results feed academic artifacts. |
| `.agents/rules/neuralops-thesis-consistency.mdc` | Thesis consistency and claims | Read before changing reported scale numbers in thesis material. |

## HPC skill and executable helpers

| File | Scaling responsibility | Change rule |
|---|---|---|
| `.agents/skills/hpc-sharif-cluster/SKILL.md` | PBS, queue, GPU allocation, environment, evidence | Mandatory before submitting GPU scale jobs. |
| `.agents/skills/hpc-sharif-cluster/README.md` | HPC quick navigation | Update only when HPC workflow entry points change. |
| `.agents/skills/hpc-sharif-cluster/scripts/run.sh` | Standard GPU job launcher | Change resource/workload arguments together; validate with `bash -n`. |
| `.agents/skills/hpc-sharif-cluster/scripts/setup_env.sh` | Remote environment setup | Change only for verified environment path/version changes. |

## Core NeuralOps engineering skills

| File | Scaling responsibility | Change rule |
|---|---|---|
| `.agents/skills/neuralops-design-coverage/SKILL.md` | Dtype, tolerance, shape, and contract scaling | Read for precision, shape, or numerical-budget changes. |
| `.agents/skills/neuralops-benchmark-auditor/SKILL.md` | Benchmark census and integrity | Read for attention/compressor/kernel breadth changes. |
| `.agents/skills/neuralops-correctness-coverage/SKILL.md` | Full correctness matrix | Read before increasing test breadth or model dimensions. |
| `.agents/skills/neuralops-cuda-kernel-debugger/SKILL.md` | CUDA numerical/debug scaling | Read for CUDA failures, OOM, NaN, Inf, or max-diff issues. |
| `.agents/skills/neuralops-dispatch-backends/SKILL.md` | CUDA/Triton/PyTorch dispatch scaling | Read when adding backend or changing fallback behavior. |
| `.agents/skills/neuralops-integration-tester/SKILL.md` | End-to-end backend integration | Read for serving, export, and multi-backend scale. |
| `.agents/skills/neuralops-kda/SKILL.md` | KDA operator scaling | Read when sequence, chunk, state, or head dimensions change. |
| `.agents/skills/neuralops-notebook-architect/SKILL.md` | Reproducible notebook scale | Read when notebook workloads or execution matrices change. |
| `.agents/skills/neuralops-nvidia-performance/SKILL.md` | NVIDIA hardware scaling/profiling | Read for GPU architecture, occupancy, VRAM, or throughput scale. |
| `.agents/skills/neuralops-optimization-loop/SKILL.md` | Iterative performance scaling | Read when changing warmup, repetitions, sweep axes, or optimization loops. |
| `.agents/skills/neuralops-perf-profiler/SKILL.md` | Nsight/performance evidence | Read for profiler-driven scale claims. |
| `.agents/skills/neuralops-scaling/SKILL.md` | Cross-file scaling contract | Mandatory entry point for every scaling request. |
| `.agents/skills/neuralops-surgery-specialist/SKILL.md` | Model surgery scale | Read when model family, layer count, heads, or healing steps change. |
| `.agents/skills/neuralops-vllm-deployment/SKILL.md` | Serving/model deployment scale | Read for model size, concurrency, KV cache, or serving throughput. |
| `.agents/skills/neuralops-ascend-performance/SKILL.md` | Ascend/NPU scale | Read only for NPU-specific scaling; keep NVIDIA and NPU evidence separate. |
| `.agents/skills/neuralops-triton-to-gluon/SKILL.md` | Triton-to-Gluon port scale | Read when porting kernels or changing NPU tile dimensions. |
| `.agents/skills/neuralops-ablation-scientist/SKILL.md` | Ablation factor scaling | Read when adding seeds, factors, regimes, or ablation axes. |
| `.agents/skills/neuralops-results-ingestion/SKILL.md` | Result ledger scaling | Read before adding, merging, or aggregating scale-point results. |
| `.agents/skills/neuralops-pr-readiness/SKILL.md` | Final quality gate | Read before declaring a scaling change complete or opening a PR. |
| `.agents/skills/neuralops-pr-stacks/SKILL.md` | Decomposing large scale changes | Read when scaling work spans multiple logical commits/PRs. |
| `.agents/skills/neuralops-release-engineer/SKILL.md` | Release/version scaling claims | Read when measured scale changes release documentation. |
| `.agents/skills/neuralops-paper-table-generator/SKILL.md` | Reproducible result tables | Read when scale results feed paper tables. |

## Academic and publication skills

All files in this section are **downstream consumers** of measured scale results. They must not invent or extrapolate measurements. Read the relevant file before changing any scale claim.

| File | Scale dimension routed | Required evidence |
|---|---|---|
| `.agents/skills/academic-humanizer/SKILL.md` | Wording of scale claims | Preserve measured/planned/failed labels. |
| `.agents/skills/bibtex-citation-provenance/SKILL.md` | Provenance of hardware/model claims | Record source and revision. |
| `.agents/skills/claim-evidence-auditor/SKILL.md` | Quantitative scale claims | Every number maps to a result artifact. |
| `.agents/skills/empirical-paper-writer/SKILL.md` | Experimental scale narratives | Use only verified matrices. |
| `.agents/skills/experimental-design/SKILL.md` | Factor/seed/regime scale | Define factors before running. |
| `.agents/skills/latex-compilation-diagnostics/SKILL.md` | Table/figure output scale | Compile after generated table changes. |
| `.agents/skills/latex-rhythm-refiner/SKILL.md` | Prose around scaling | Do not change numeric evidence. |
| `.agents/skills/paper-abstract-intro-architect/SKILL.md` | High-level scale claims | Cite evidence; avoid unsupported generalization. |
| `.agents/skills/paper-arxiv-submission-packager/SKILL.md` | Camera-ready scale artifacts | Package exact verified outputs. |
| `.agents/skills/paper-attention-mechanisms-taxonomy/SKILL.md` | Attention breadth scale | Preserve canonical registry counts. |
| `.agents/skills/paper-codebase-deep-explorer/SKILL.md` | Cross-file dependency scale | Rebuild maps when ownership changes. |
| `.agents/skills/paper-cuda-cpp-microarchitecture/SKILL.md` | CUDA hardware scale | Tie claims to actual architecture/job. |
| `.agents/skills/paper-empirical-evaluator/SKILL.md` | Benchmark matrix scale | Use raw telemetry and manifests. |
| `.agents/skills/paper-fisher-information-spectral/SKILL.md` | Spectral/dimension scale | Record dimensions and precision. |
| `.agents/skills/paper-hessian-eigenspectrum-loss/SKILL.md` | Dataset/model scale | Record sample/model configuration. |
| `.agents/skills/paper-kernel-postmortem-forensics/SKILL.md` | Failure scale/postmortem | Preserve negative results and root causes. |
| `.agents/skills/paper-knowledge-distillation-healing/SKILL.md` | Distillation/healing scale | Record steps, seeds, and model sizes. |
| `.agents/skills/paper-kv-cache-eviction-policies/SKILL.md` | Context/KV scale | Record sequence, cache, and policy. |
| `.agents/skills/paper-linear-recurrent-associative-scans/SKILL.md` | Sequence/chunk scale | Record scan length and chunk size. |
| `.agents/skills/paper-lowrank-svd-factorization/SKILL.md` | Rank/compression scale | Record rank, ratio, and reconstruction error. |
| `.agents/skills/paper-math-formulation-writer/SKILL.md` | Mathematical dimension scale | Keep formulas consistent with implementation. |
| `.agents/skills/paper-multi-seed-statistical-rigor/SKILL.md` | Seed/sample scale | Report seed set and confidence intervals. |
| `.agents/skills/paper-riemannian-stiefel-optimizer/SKILL.md` | Optimizer/dimension scale | Record rank, steps, and tolerance. |
| `.agents/skills/paper-roofline-hardware-profiler/SKILL.md` | Hardware/sequence scale | Record clocks, memory, and arithmetic intensity. |
| `.agents/skills/paper-serving-radix-paged-attention/SKILL.md` | Concurrency/KV scale | Verify real engine integration. |
| `.agents/skills/paper-structured-pruning-admm/SKILL.md` | Sparsity/channel scale | Record pruning ratio and hardware format. |
| `.agents/skills/paper-systems-kernel-architect/SKILL.md` | System/backend scale | Preserve the 3-level dispatch contract. |
| `.agents/skills/paper-tensor-quantization-microscaling/SKILL.md` | Quantization/dtype scale | Record format, group size, and error. |
| `.agents/skills/paper-triton-kernel-architect/SKILL.md` | Triton tile/grid scale | Record kernel version and parity. |
| `.agents/skills/neuralops-presentation-architect/SKILL.md` | Presentation charts and scale claims | Regenerate from verified tables; preserve failed/skipped states. |
| `.agents/skills/roofline-visualizer-generator/SKILL.md` | Roofline axes/points | Generate from measured telemetry only. |
| `.agents/skills/scientific-visualization/SKILL.md` | Plot dimensions/series | Keep failed/skipped points visible. |
| `.agents/skills/scientific-writing/SKILL.md` | Scientific scale wording | Distinguish measurement from projection. |

## Individual reference and script files

These supporting files are not interchangeable. Read the exact file when its parent skill is selected.

| File | Scale routing |
|---|---|
| `.agents/skills/neuralops-ascend-performance/references/TRAPS.md` | NPU trap/failure evidence; update only with a verified trap. |
| `.agents/skills/neuralops-ascend-performance/references/cases.md` | NPU case-study scale; preserve hardware and workload identifiers. |
| `.agents/skills/neuralops-ascend-performance/references/g-contiguous-loading.md` | NPU memory-layout scale; verify contiguous dimensions. |
| `.agents/skills/neuralops-ascend-performance/references/reference.md` | NPU reference contract; no unsupported extrapolation. |
| `.agents/skills/neuralops-ascend-performance/scripts/analyze_profile.py` | Profile aggregation; input/output schema must include scale axes. |
| `.agents/skills/neuralops-ascend-performance/scripts/profile_npu.py` | NPU profiling workload; change dimensions with the manifest. |
| `.agents/skills/neuralops-correctness-coverage/references/cp.md` | Correctness case; preserve tolerance and shape evidence. |
| `.agents/skills/neuralops-correctness-coverage/references/delta-rule.md` | Delta-rule sequence/state scale; record length and dtype. |
| `.agents/skills/neuralops-correctness-coverage/references/generalized-delta-rule.md` | Generalized delta dimensions; verify parity. |
| `.agents/skills/neuralops-correctness-coverage/references/simple-gla.md` | GLA dimensions; verify causal and numerical gates. |
| `.agents/skills/neuralops-optimization-loop/references/TRAPS.md` | Optimization failure modes; never hide scale failures. |
| `.agents/skills/neuralops-optimization-loop/references/opt-log-template.md` | Optimization log schema; every run needs scale metadata. |

No other `.agents` file is outside this matrix. If `find .agents -type f` reports a new path, add it here before using it for a scaling task.

## File-level routing protocol

For each requested scale change, create a temporary checklist with one row per target file:

```text
path | read completely | caller/dependent search | intended edit | verification | evidence path
```

Do not mark the task complete until every selected row has a result. If a file is inspected and intentionally unchanged, record `read-only — no ownership change`; do not modify it merely to create activity. The project’s source files, tests, manifests, and docs remain the authoritative implementation surface; `.agents` routes the work but does not replace those files.

