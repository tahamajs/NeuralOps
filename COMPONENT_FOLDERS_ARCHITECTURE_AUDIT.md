# NeuralOps Component Folders: Complete Architecture Audit and Integration Plan

## Scope

This document audits the component folders shown in the project tree and maps
how they should connect:

```text
inference
integrations
kernels
layers
models
operations
ops
optimization
pipeline_transform
recipe
reporting
results_saver
runners
```

The purpose is to remove architectural ambiguity. Each folder should have one
clear responsibility, one public entry boundary, and explicit data contracts to
the folders around it.

## Executive finding

The repository already contains most of the required capabilities, but they are
not yet organized as one linear product pipeline. There are several overlapping
concepts:

- `ops/`, `kernels/`, and parts of `layers/` all look like kernel owners;
- `results_saver/`, `reporting/`, benchmark result writers, and runner writers
  all look like result owners;
- `runners/`, benchmark orchestrators, pipeline executors, and CLI handlers all
  look like workflow owners;
- `recipe/`, `pipeline_transform/`, and `operations/` all describe how a model
  is changed;
- `inference/`, `models/`, and `integrations/` all contain runtime/model
  execution concerns.

The cleanup should not delete these capabilities. It should establish a strict
direction of dependencies and adapters between them.

## 1. Target dependency graph

The desired high-level flow is:

```text
CLI / API
   |
   v
recipe.schema + recipe.loader
   |
   v
operations (what to change)
   |
   v
pipeline_transform (how to plan/apply/rollback)
   |
   +----------------------+
   v                      v
models                layers
   |                      |
   +-----------> ops/kernels
                      |
                      v
                 inference
                      |
                      v
               integrations
                      |
                      v
             reporting/results_saver
                      |
                      v
                    runners
```

`runners` orchestrates the whole operation but must not own implementation
logic. It calls the public interfaces of the other folders.

## 2. Dependency rules

### Allowed direction

```text
recipe -> operations -> pipeline_transform -> models/layers/ops
models -> layers -> ops
inference -> models/ops/integrations
reporting -> all read-only metadata/result contracts
results_saver -> reporting/result contracts
runners -> all public package APIs
```

### Forbidden direction

- `layers` must not import `runners`.
- `ops` must not import CLI code, reporting, or result persistence.
- `results_saver` must not execute model transformations.
- `reporting` must not mutate model weights.
- `integrations` must not reimplement the registry.
- `pipeline_transform` must not write directly to arbitrary result files.
- `runners` must not reach into private implementation symbols when a public
  adapter exists.

## 3. Folder-by-folder audit

## 3.1 `neuralops/inference/`

### Current role

This folder contains runtime execution components:

```text
engine.py
paged_cache.py
pipeline_parallel.py
speculative_decoding.py
```

### It should own

- prefill and decode execution;
- request batching and scheduling;
- KV-cache lifecycle;
- speculative decoding control flow;
- pipeline-parallel inference coordination.

### It should not own

- model surgery or compression algorithms;
- registry discovery;
- benchmark result serialization;
- vendor-specific serving integration code.

### Required connections

```text
models -> inference.engine
inference.engine -> inference.paged_cache
inference.engine -> inference.speculative_decoding
inference.engine -> integrations.serving adapters
inference.engine -> reporting telemetry
```

### Problems to audit

1. Device and dtype selection must come from one runtime config, not be inferred
   independently by engine and cache.
2. KV-cache shape/layout must be declared in a protocol shared with vLLM,
   SGLang, and TensorRT-LLM adapters.
3. Speculative decoding must report accepted/rejected draft tokens rather than
   only final latency.
4. Pipeline parallel failures must identify stage and rank.
5. Cache allocation failures must be visible as `FAIL_RESOURCE`, not converted
   to an empty cache or a CPU path.

### Recommended public API

```python
class InferenceRequest:
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    sampling: SamplingConfig


class InferenceResult:
    request_id: str
    output_ids: torch.Tensor
    ttft_ms: float
    itl_ms: float
    accepted_draft_tokens: int
    backend: str


class InferenceEngine:
    def generate(self, request: InferenceRequest) -> InferenceResult: ...
```

## 3.2 `neuralops/integrations/`

### Current role

This folder contains adapters for external ecosystems:

```text
hf_automodel.py
vllm.py / vllm/
sglang.py / sglang/
tensorrt_llm.py / tensorrt_llm_plugin/
tgi.py
serving.py
compressed_tensors.py
distributed_compile.py
fla/
```

### It should own

- external API adapters;
- conversion between NeuralOps contracts and vendor contracts;
- optional dependency detection;
- version compatibility checks;
- external cache/engine metadata translation.

### It should not own

- canonical attention/compressor registries;
- core numerical implementations;
- benchmark pass/fail policy;
- silent fallback to another vendor.

### Design requirement: capability adapters

Each integration should expose a capability object rather than forcing callers
to import vendor internals:

```python
@dataclass(frozen=True)
class IntegrationCapabilities:
    name: str
    version: str | None
    available: bool
    supports_cuda: bool
    supports_streaming: bool
    supports_paged_kv: bool
    reason_unavailable: str | None = None


class ServingIntegration(Protocol):
    def capabilities(self) -> IntegrationCapabilities: ...
    def load(self, config: ServingConfig) -> ServingBackend: ...
```

### Problems to audit

1. Optional imports must not happen at package import time.
2. An unavailable vLLM/SGLang/TensorRT-LLM dependency must produce an explicit
   unavailable capability, not a fake benchmark row.
3. All adapters must use the same `ServingConfig` and `KVCacheSpec`.
4. Version checks must be centralized and testable.
5. External exceptions must preserve the original error and integration name.

## 3.3 `neuralops/kernels/`

### Current role

This is a high-level kernel namespace containing `extended_kernels.py` and
backend package markers.

### Recommended role

Make this folder a public kernel facade only:

```text
neuralops.kernels
  -> public operator names
  -> dispatch to ops/cuda, ops/triton, ops/pytorch
```

It should not contain a second implementation of a kernel already owned by
`ops/`.

### Problems to audit

- duplicate symbols between `kernels/extended_kernels.py` and `ops/`;
- unclear distinction between public operator and backend implementation;
- backend imports that trigger JIT compilation during import;
- no unified metadata describing actual backend used.

### Target API

```python
from neuralops.kernels import dispatch_kernel

result = dispatch_kernel(
    "selective_scan",
    inputs,
    backend="auto",
    strict=True,
)
```

The returned object should include `output`, `backend_used`, `fallbacks`, and
`max_diff` when a reference parity check was requested.

## 3.4 `neuralops/layers/`

### Current role

This is the canonical module implementation layer for attention mechanisms and
related neural layers. It contains families such as `linear`, `gated`,
`hybrid`, `modern`, `sparse`, `ssm`, `svd`, and `fast_kernels`.

### It should own

- `torch.nn.Module` implementations;
- shape/dtype/causal contracts;
- differentiable forward and backward behavior;
- module-level use of kernel facades.

### It should not own

- benchmark orchestration;
- direct CSV/Parquet writing;
- vendor serving adapters;
- model checkpoint transaction logic.

### Every layer should expose

```python
class AttentionContract:
    name: str
    supports_causal: bool
    supports_gqa: bool
    supported_dtypes: tuple[torch.dtype, ...]
    backend_preferences: tuple[str, ...]


class MyAttention(nn.Module):
    contract: AttentionContract

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
        **kwargs: object,
    ) -> torch.Tensor: ...
```

### Problems to audit

1. Some layers contain backend selection logic that should move to kernel
   dispatch.
2. Causal-mask semantics need a shared test contract.
3. Input aliases (`dim`, `hidden_size`, `d_model`) should be normalized by the
   registry factory, not reimplemented in every class.
4. Triton or CUDA failures must not be swallowed inside a layer.
5. Every layer must have a reference implementation path for correctness tests.

## 3.5 `neuralops/models/`

### Current role

This folder contains model wrappers, configuration, model loading, CUDA graph,
cache, RoPE, hybrid blocks, and surgical I/O.

### It should own

- model architecture composition;
- loading and saving model checkpoints;
- model-level configuration;
- model-to-layer wiring;
- model-level cache and graph state.

### It should not own

- compression algorithm implementations;
- benchmark timing utilities;
- vendor engine adapters.

### Required connections

```text
models.hub -> integrations.hf_automodel
models.modeling -> layers + kernels
models.surgical_io -> operations.export + recipe manifests
models.cache/paged_state -> inference.paged_cache
models.configuration -> recipe.schema
```

### Critical design rule

Model loading and model mutation must be separate transactions:

```text
load -> validate -> plan operation -> apply to copy/snapshot -> verify -> commit
```

Never mutate a live model before the operation plan and rollback snapshot are
created.

## 3.6 `neuralops/operations/`

### Current role

This folder describes transformations such as attention replacement,
compression, pruning, quantization, LoRA, distillation, healing, and export.

### It should own

- operation definitions;
- operation validation;
- operation inputs and outputs;
- composable transformation semantics.

### It should not own

- scheduling or worker pools;
- low-level CUDA kernels;
- arbitrary checkpoint writes;
- CLI parsing.

### Operation contract

```python
class Operation(Protocol):
    name: str

    def validate(self, context: OperationContext) -> ValidationReport: ...
    def plan(self, context: OperationContext) -> OperationPlan: ...
    def apply(self, plan: OperationPlan) -> OperationResult: ...
    def rollback(self, result: OperationResult) -> None: ...
```

The current subfolders map naturally:

```text
operations/attention   -> attention replacement/configuration
operations/weights     -> compression/pruning/quantization/decomposition
operations/adaptation  -> LoRA/KD/healing
operations/architecture-> structural model changes
operations/export      -> validated checkpoint/export artifacts
operations/verify      -> parity and contract checks
```

## 3.7 `neuralops/ops/`

### Current role

This is the backend implementation layer:

```text
ops/cuda
ops/triton
ops/pytorch
```

### It should own

- native CUDA/HIP extension loading;
- Triton kernels and autotuning;
- PyTorch mathematical reference kernels;
- one fail-closed dispatcher.

### Required backend contract

```python
class KernelAttempt:
    backend: Literal["cuda", "triton", "pytorch"]
    status: Literal["success", "failed", "unavailable"]
    error: str | None


class DispatchResult:
    output: torch.Tensor
    backend_used: str
    attempts: list[KernelAttempt]
    max_diff: float | None
```

The dispatcher must emit a warning for every fallback and must never report
`backend_used="cuda"` when the output came from Triton or PyTorch.

### Folder-specific placement

```text
ops/cuda/<name>/       native source, binding, loader, autograd wrapper
ops/triton/<name>.py   Triton implementation and reference comparison
ops/pytorch/<name>.py  mathematical oracle
ops/_dispatch.py       backend selection and telemetry
```

Empty CUDA directories are not acceptable as silent packages. They must either
contain a valid loader that raises an informative error or be represented in a
manifest as unavailable.

## 3.8 `neuralops/optimization/`

### Current role

This folder currently appears small, with a kernels namespace. It should not
become a second home for training or compression algorithms without a clear
boundary.

### Recommended role

Reserve it for optimization algorithms and optimization-specific kernels:

```text
optimization/
  optimizers/       parameter update algorithms
  schedulers/       LR and rank schedules
  objectives/       KD/healing/compression objectives
  kernels/          optimization-only fused kernels
```

Do not put generic CUDA kernels here; those belong under `ops/cuda`.

## 3.9 `neuralops/pipeline_transform/`

### Current role

This folder contains planning, dependency graphs, execution, hardware planning,
rollback, snapshots, and results.

### It should own

- immutable transformation plans;
- dependency ordering;
- resource planning;
- transactional execution;
- rollback and snapshot lifecycle.

### It should not own

- model-specific attention implementations;
- CLI parsing;
- result ledger persistence.

### Required flow

```text
recipe -> planner -> Plan -> dependency validation -> hardware planner
       -> executor -> snapshot -> operation apply -> verification
       -> commit or rollback -> OperationResult
```

The executor should return structured events that `reporting` can consume.

## 3.10 `neuralops/recipe/`

### Current role

This folder contains canonical recipe schema, loading, serialization, errors,
and versioning.

### It should be the declarative source of truth

Recipes should describe intent, not implementation details:

```yaml
version: 1
model:
  source: distilbert-base-uncased
operations:
  - type: replace_attention
    target: encoder.layer.*
    attention: gated_delta_net
    parameters:
      rank: 64
validation:
  parity_tolerance: 0.001
output:
  directory: results/recipe_run
```

The recipe loader must validate schema, resolve aliases, calculate a content
hash, and never execute code while parsing.

## 3.11 `neuralops/reporting/`

### Current role

This folder contains environment capture, evidence, and report generation.

### It should own

- execution evidence;
- environment metadata;
- human reports;
- machine-readable summaries.

### It should consume

```text
OperationResult
DispatchResult
BenchmarkResult
EnvironmentSnapshot
```

It must not reach into a model to infer undocumented metrics. Every reported
number must carry its source run ID and metric name.

## 3.12 `neuralops/results_saver/`

### Current role

This folder contains saving configuration, writers, storage, plotting, and
Hugging Face backends.

### Recommended responsibility

Make this the single persistence layer for completed result records. The
benchmark-specific `results/` package and this folder currently overlap and
must be reconciled.

Recommended split:

```text
results/schema.py      canonical dataclasses and validation
results_saver/         persistence backends and transactional writers
reporting/             human-readable reports and evidence
```

The saver should provide:

```python
class ResultsStore(Protocol):
    def write(self, record: ResultRecord) -> None: ...
    def validate(self, path: Path) -> ValidationReport: ...
    def compare(self, baseline: Path, candidate: Path) -> Comparison: ...
```

Parquet, SQLite, JSONL, and CSV are output backends—not separate schemas.

## 3.13 `neuralops/runners/`

### Current role

This folder contains master pipeline execution, sweep execution, monitoring,
telemetry, and orchestration.

### It should own

- scheduling multiple operations or benchmarks;
- worker lifecycle;
- resume and monitoring;
- aggregation of structured results.

### It should not own

- module implementation;
- low-level device checks duplicated from `ops`;
- direct writes to arbitrary CSV files.

### Runner contract

```python
class RunRequest:
    run_id: str
    recipe: Recipe
    config: RunConfig
    output_dir: Path


class RunOrchestrator:
    def plan(self, request: RunRequest) -> ExecutionPlan: ...
    def execute(self, plan: ExecutionPlan) -> RunResult: ...
    def resume(self, run_id: str) -> RunResult: ...
```

Monitoring should read event streams or result records. It must not inspect
private worker state or change benchmark outcomes.

## 4. Cross-folder data contracts

### 4.1 Configuration contract

One canonical config should feed CLI, recipe execution, inference, kernels,
benchmarking, and saving:

```python
@dataclass
class RuntimeConfig:
    device: str
    precision: str
    seed: int
    output_dir: Path
    strict_backend: bool
    parity_tolerance: float
```

Folder-specific config should be nested under this runtime config rather than
redeclaring `device`, `dtype`, and `output_dir` in every folder.

### 4.2 Event contract

All long operations should emit events:

```python
@dataclass
class RunEvent:
    run_id: str
    timestamp: str
    component: str
    event: str
    status: str
    payload: dict[str, object]
```

Examples:

```text
operation.started
kernel.fallback
parity.completed
checkpoint.written
run.failed
```

### 4.3 Artifact contract

Every run directory should contain:

```text
config.json
environment.json
manifest.json
events.jsonl
results.jsonl
summary.json
logs/
artifacts/
```

No folder should invent a competing `latest.json` format.

## 5. Main current problems to fix

1. **Overlapping owners:** kernels, results, runners, and transformations each
   have multiple possible owners.
2. **Private cross-imports:** callers can bypass stable APIs and import internal
   classes directly.
3. **Duplicate result writers:** benchmark results and `results_saver` can emit
   different schemas.
4. **No universal run context:** device, seed, git commit, and output path are
   not guaranteed to travel across all folders.
5. **Backend identity loss:** fallback output can be labelled as the requested
   backend rather than the actual backend.
6. **Mutation without transaction:** model operations need plan, snapshot,
   verify, commit, and rollback boundaries.
7. **Optional integrations at import time:** missing vendor packages can make
   unrelated commands fail during import.
8. **Generated files inside source folders:** `.DS_Store` and `__pycache__`
   indicate source-tree hygiene issues.
9. **Empty namespaces:** folders such as optimization need an explicit scope so
   they do not become dumping grounds.
10. **Monitoring coupled to execution internals:** monitoring should consume
    stable events, not inspect implementation state.
11. **Recipe and CLI duplication:** both can describe operations, but their
    semantics must be merged into one validated request.
12. **No complete integration test:** individual pieces may pass while the full
    chain from recipe to report is broken.

## 6. Recommended implementation order

### Phase 1: contracts

1. Define `RuntimeConfig`, `RunContext`, `RunEvent`, and `ResultRecord`.
2. Add schema validation and serialization tests.
3. Make device, dtype, seed, and output directory mandatory in the context.

### Phase 2: backend boundary

1. Make `kernels` a facade.
2. Move backend selection into one dispatcher under `ops`.
3. Add explicit fallback telemetry.
4. Add parity and gradient gates.

### Phase 3: transformation boundary

1. Make recipes declarative.
2. Make operations validate and produce plans.
3. Make pipeline transforms transactional.
4. Connect snapshots and rollback to operation results.

### Phase 4: persistence boundary

1. Select one canonical result schema.
2. Make `results_saver` the persistence backend layer.
3. Make reporting consume records rather than raw model state.
4. Add SQLite/JSONL/CSV/Parquet backends behind one interface.

### Phase 5: runtime boundary

1. Connect models to layers and kernel facades.
2. Connect inference to model runtime and cache contracts.
3. Connect external integrations through capability adapters.
4. Add serving integration tests.

### Phase 6: orchestration

1. Make runners consume recipes and runtime configs.
2. Add resumable run directories and locks.
3. Connect monitoring to events.
4. Add full end-to-end tests.

## 7. Required tests

```text
tests/architecture/test_import_boundaries.py
  ops does not import runners or reporting
  layers does not import CLI
  results_saver does not mutate models

tests/architecture/test_contracts.py
  RuntimeConfig roundtrip
  RunEvent roundtrip
  ResultRecord schema validation

tests/architecture/test_recipe_pipeline.py
  recipe -> plan -> execute -> report
  invalid recipe fails before mutation
  failed operation rolls back snapshot

tests/architecture/test_dispatch_integration.py
  backend identity is preserved
  fallback warnings are emitted
  numerical parity gate is enforced

tests/architecture/test_inference_integrations.py
  serving capability discovery
  optional dependency unavailable state
  KV-cache spec translation

tests/architecture/test_results_pipeline.py
  JSONL/CSV/SQLite consistency
  Parquet missing dependency policy
  duplicate run rejection
```

## 8. Practical placement rules

When adding a new file, use this decision table:

| New functionality | Correct location |
|---|---|
| New attention `nn.Module` | `layers/<family>/` |
| New compressor | `compression/<family>/` |
| CUDA/Triton/PyTorch operator backend | `ops/<backend>/` |
| Public kernel facade | `kernels/` |
| Model wiring/configuration | `models/` |
| Model mutation | `operations/` |
| Declarative model change | `recipe/` |
| Plan/snapshot/rollback | `pipeline_transform/` |
| Inference request/cache/decode | `inference/` |
| vLLM/SGLang/TensorRT adapter | `integrations/` |
| Human report/evidence | `reporting/` |
| Result persistence backend | `results_saver/` |
| Multi-step scheduling/monitoring | `runners/` |
| Optimizer or optimization objective | `optimization/` |

## 9. Definition of done

The folders are correctly connected when:

- a recipe can be loaded without executing code;
- the recipe becomes a validated operation plan;
- the plan loads a model and applies layers through public APIs;
- layers call the kernel facade, not vendor internals;
- dispatch reports the backend actually used;
- inference can consume the resulting model;
- integrations expose capabilities and adapt external runtimes;
- reporting receives structured events and results;
- results_saver persists one canonical schema;
- runners can resume without duplicate writes;
- import-boundary tests pass;
- no folder has an undocumented second owner;
- generated files are excluded from source packages;
- end-to-end tests prove recipe -> operation -> runtime -> report.

Until these gates pass, the architecture should be treated as transitional.

## 10. Exact keep / move / delete policy

This section is intentionally concrete. It is the checklist to use before
moving any file. A file should not be deleted merely because another folder has
a similar name; first redirect callers, add a compatibility import if needed,
and run the boundary tests.

### 10.1 Keep as canonical implementation

Keep these responsibilities in their current conceptual owners:

```text
layers/                 torch.nn.Module implementations
ops/cuda/               native CUDA loaders and bindings
ops/triton/             Triton implementations
ops/pytorch/            reference implementations
models/                 model composition and checkpoint lifecycle
operations/             model mutations and validations
pipeline_transform/     plans, dependency graphs, snapshots, rollback
recipe/                 declarative schemas and serialization
inference/              runtime requests, cache, decoding
integrations/           external runtime adapters
reporting/              evidence and human reports
results_saver/          persistence backends
runners/                scheduling and monitoring
```

### 10.2 Move or adapt

The following concepts should be consolidated:

| Current overlap | Canonical owner | Required action |
|---|---|---|
| `kernels/extended_kernels.py` and `ops/*` | `ops/` implementation + `kernels/` facade | Move implementation details to `ops`; keep re-export facade |
| benchmark writers and `results_saver/` | `results_saver/` | Make benchmark writers call `ResultsStore` |
| benchmark orchestrators and `runners/` | `runners/` | Keep benchmark-specific plan builders, centralize scheduling |
| recipe execution and CLI handlers | `recipe/` + `pipeline_transform/` | CLI parses only; loader/planner executes |
| serving benchmark and `inference/` | `inference/` | Benchmark measures engine through public inference API |
| vendor checks and integrations | `integrations/` | Expose capabilities instead of scattered imports |
| model surgery and direct model writes | `operations/` + `models/surgical_io.py` | Use operation transaction and checkpoint writer |
| dispatch statistics and reporting | `ops` emits events; `reporting` consumes | Remove direct reporting imports from kernels |
```

### 10.3 Remove after migration

These are valid deletion candidates only after caller verification:

- duplicate result serializers that emit a different schema;
- wrapper scripts that contain copied benchmark logic;
- unused `.DS_Store` files;
- tracked `__pycache__` or generated bytecode;
- empty backend package directories that do not expose a loader or manifest;
- duplicate compatibility modules with no imports or documentation references;
- direct vendor imports in core code when an integration adapter exists.

Before deletion, run:

```bash
rg -n "from neuralops\.|import neuralops\." .
rg -n "old_module_name|old_symbol_name" neuralops tests benchmarks scripts
git grep -n "path/to/file.py"
```

Record the result in the migration matrix.

## 11. Exact connection map

### 11.1 CLI to recipe/config

```text
neuralops CLI
  -> parse only command-line syntax
  -> ConfigLoader.load()
  -> RuntimeConfig
  -> RecipeLoader.load() if --recipe is used
  -> RunRequest
```

The CLI must not instantiate an attention, open a checkpoint, or call a CUDA
kernel directly. It should create a request and pass it to a command handler.

Suggested code:

```python
def run_command(args: argparse.Namespace) -> int:
    runtime = ConfigLoader.from_cli(args)
    recipe = RecipeLoader.load(args.recipe) if args.recipe else None
    request = RunRequest(runtime=runtime, recipe=recipe)
    result = RunOrchestrator().execute(request)
    ResultsStore.from_config(runtime.results).write(result)
    return result.exit_code
```

### 11.2 Recipe to operation

```text
recipe/schema.py
  -> validated Recipe
recipe/canonical.py
  -> normalized operation names and aliases
recipe/loader.py
  -> immutable RecipeDocument
operations/registry.py
  -> OperationFactory.create(document.operation)
operations/*
  -> Operation.validate(context)
  -> Operation.plan(context)
```

The recipe must not contain Python import paths that are executed dynamically.
Use a registry key, then resolve that key through the existing registry.

Example:

```yaml
operations:
  - type: replace_attention
    target: encoder.layer.*
    attention: gated_delta_net
    params:
      causal: true
```

The loader resolves `replace_attention` to an operation factory and
`gated_delta_net` to a registered layer. It does not import an arbitrary module
named by the user.

### 11.3 Operation to model

```text
OperationContext
  -> ModelHandle
  -> registry lookup
  -> capability validation
  -> snapshot
  -> operation apply
  -> model validation
```

The model boundary should expose:

```python
class ModelHandle(Protocol):
    def named_modules(self) -> Iterator[tuple[str, nn.Module]]: ...
    def snapshot(self) -> Snapshot: ...
    def restore(self, snapshot: Snapshot) -> None: ...
    def save(self, destination: Path) -> ArtifactManifest: ...
```

Operations must not manipulate checkpoint paths directly. They call
`ModelHandle.save()` and receive a manifest.

### 11.4 Layers to kernel facade

```text
layers/gated/gated_delta_net.py
  -> kernels.dispatch("gdn_scan", ...)
  -> ops._dispatch
      -> ops.cuda.gdn_scan
      -> ops.triton.gdn_scan
      -> ops.pytorch.gdn_scan
```

The layer owns tensor semantics and shape checks. The dispatcher owns backend
choice. This prevents every layer from reimplementing CUDA/Triton fallback.

Layer example:

```python
def forward(self, q, k, v, *, is_causal=False):
    validate_attention_inputs(q, k, v)
    return dispatch_kernel(
        "gdn_scan",
        q=q,
        k=k,
        v=v,
        is_causal=is_causal,
        runtime=self.runtime,
    ).output
```

### 11.5 Dispatcher to reporting

The dispatcher should return a structured attempt list and optionally emit a
`kernel.fallback` event. It must not write JSON or import reporting writers.

```python
DispatchResult(
    output=output,
    backend_used="triton",
    attempts=[
        KernelAttempt("cuda", "failed", "extension not found"),
        KernelAttempt("triton", "success", None),
    ],
    max_diff=3.1e-4,
)
```

The caller converts this to a result record:

```python
record.metrics["max_diff"] = dispatch.max_diff
record.metadata["backend_used"] = dispatch.backend_used
record.metadata["attempts"] = [asdict(item) for item in dispatch.attempts]
```

### 11.6 Model to inference

```text
models.modeling.Model
  -> inference.engine.load_model(model_handle)
  -> inference request
  -> cache allocation
  -> prefill/decode
  -> InferenceResult
```

Inference must receive an already validated model handle. It should not call
model surgery or compression itself.

### 11.7 Inference to integrations

```text
InferenceEngine
  -> ServingIntegrationRegistry.get("vllm")
  -> capabilities()
  -> load(ServingConfig)
  -> ServingBackend
```

The engine should operate against a backend protocol. Vendor adapters translate
requests and cache specs but do not modify core inference contracts.

### 11.8 Results to reporting

```text
RunEvent + ResultRecord
  -> ResultsStore.write()
  -> reporting.evidence.collect()
  -> reporting.report.render()
```

Reporting should never scrape log text to calculate metrics when structured
records are available. Logs are evidence of execution; results records are the
source of numerical claims.

### 11.9 Reporting to runners

```text
Runner emits events
  -> EventSink
  -> JSONL event stream
  -> monitor / report / UI subscribers
```

The monitor must be a subscriber. It must not import private fields from a
worker or mutate its state.

## 12. Exact file-level changes recommended

### 12.1 `neuralops/inference`

Add:

```text
contracts.py       InferenceRequest, InferenceResult, SamplingConfig
backend.py         InferenceBackend protocol
telemetry.py       TTFT/ITL/token counters
```

Modify:

- `engine.py` to accept `RuntimeConfig` and `ModelHandle`;
- `paged_cache.py` to consume one `KVCacheSpec` from integrations;
- `speculative_decoding.py` to return accepted/rejected token counts;
- `pipeline_parallel.py` to emit rank/stage errors.

Remove or relocate:

- vendor-specific code from `engine.py` to `integrations/`;
- result CSV writing from inference modules;
- duplicated device selection helpers.

### 12.2 `neuralops/integrations`

Add:

```text
base.py           IntegrationCapabilities and protocols
registry.py       optional integration registry
errors.py         IntegrationUnavailable, VersionMismatch
```

Modify each vendor adapter to implement `capabilities()` and `load()`.

Remove:

- unconditional imports of `vllm`, `sglang`, `tensorrt_llm`, or TGI;
- fallback to a different vendor without a recorded decision;
- benchmark-specific metric calculation from adapters.

### 12.3 `neuralops/kernels`

Keep only public facade exports. Move implementation details that duplicate
`ops/cuda`, `ops/triton`, or `ops/pytorch` to their canonical backend.

Add:

```text
contracts.py       KernelRequest, DispatchResult, KernelAttempt
dispatch.py        public dispatch_kernel facade
```

### 12.4 `neuralops/layers`

Add shared validation modules:

```text
contracts.py       input shape/dtype/causal contracts
validation.py      finite, gradient, causal checks
```

Modify each family package so its modules call `kernels.dispatch_kernel()`.

Remove local copies of backend fallback logic once equivalent tests exist.

### 12.5 `neuralops/models`

Add:

```text
protocols.py       ModelHandle and checkpoint protocols
manifests.py       model artifact manifest schema
```

Modify `surgical_io.py` to use atomic output directories and manifests.
Move operation-specific mutation logic out of `modeling.py` into
`operations/`.

### 12.6 `neuralops/operations`

Add:

```text
context.py         OperationContext
plan.py            OperationPlan
transaction.py     snapshot/apply/verify/commit/rollback
```

Modify every operation to implement `validate`, `plan`, `apply`, and
`rollback`. Remove direct CLI argument parsing from operation modules.

### 12.7 `neuralops/ops`

Add or consolidate:

```text
contracts.py       backend contracts
dispatch.py        one dispatcher
telemetry.py       backend attempt events
parity.py          reference comparison
```

Ensure CUDA, Triton, and PyTorch implementations all share operator names and
return shapes. A missing native implementation must be represented explicitly,
not as an empty package.

### 12.8 `neuralops/optimization`

Create a clear substructure before adding more algorithms:

```text
optimizers/
schedulers/
objectives/
kernels/
```

Move only optimization-specific code here. Do not move generic backend kernels.

### 12.9 `neuralops/pipeline_transform`

Make `plan.py`, `executor.py`, `snapshot.py`, and `rollback.py` the only public
transaction path. Add integration tests that prove an exception restores the
original model state bit-for-bit where feasible.

### 12.10 `neuralops/recipe`

Add schema validation for operation types, target selectors, backend policy,
parity tolerance, and output policy. Add a recipe hash to every run manifest.

### 12.11 `neuralops/reporting`

Add report adapters for `RunEvent`, `DispatchResult`, `OperationResult`, and
`BenchmarkResult`. Do not infer missing values. Render unavailable values as
`null` plus a reason, or fail validation when the metric is required.

### 12.12 `neuralops/results_saver`

Make `ResultsStore` the only writer used by new code. Add schema versioning,
atomic writes, duplicate detection, and explicit optional-format policy.

### 12.13 `neuralops/runners`

Add a `RunRequest` boundary and make the master orchestrator consume plans and
events. Move monitoring to event subscriptions. Add run locks and resume
validation.

## 13. What should be deleted or forbidden

The following patterns should be removed from all these folders:

```text
except: pass
except Exception: return None       # in data-producing paths
hard-coded metrics or baselines
fake success records after backend failure
generic SVD under another compressor name
direct CPU fallback after explicit CUDA
fixed results/latest.json writes
duplicate device/dtype parsers
vendor imports at module import time
private imports across folder boundaries
direct result writes from layers or kernels
unvalidated arbitrary Python paths in recipes
untracked .DS_Store and __pycache__ files
```

The following should be allowed only with explicit metadata:

```text
CUDA -> Triton fallback
Triton -> PyTorch fallback
optional vendor unavailable
Parquet unavailable while JSON/CSV succeeds
unsupported dtype or causal mode
```

## 14. End-to-end example after cleanup

### Input recipe

```yaml
version: 1
model:
  source: distilbert-base-uncased
runtime:
  device: cuda:0
  precision: float16
  seed: 42
  strict_backend: false
operations:
  - type: replace_attention
    target: encoder.layer.*
    attention: gated_delta_net
    params:
      rank: 64
validation:
  gates: [forward, backward, finite, causal, parity]
  parity_tolerance: 0.001
output:
  directory: results/example_run
```

### Exact flow

```text
CLI parses path
  -> RecipeLoader validates schema
  -> hash recipe
  -> ConfigLoader resolves cuda:0 and float16
  -> Runner creates results/example_run/.active.lock
  -> Planner resolves replace_attention operation
  -> ModelLoader loads model
  -> Snapshot captures original state
  -> Operation validates gated_delta_net capability
  -> Operation replaces target layers
  -> Layer calls kernels.dispatch_kernel("gdn_scan")
  -> Dispatcher tries CUDA, then Triton, then PyTorch if policy allows
  -> Validation compares output against PyTorch reference
  -> Reporter receives structured gate events
  -> ResultsStore writes typed records atomically
  -> Model artifact is saved with manifest
  -> Runner removes lock and writes summary
```

### Failure flow

```text
CUDA extension fails
  -> warning: level=1 error=...
  -> Triton fails
  -> warning: level=2 error=...
  -> PyTorch executes
  -> record backend_used=pytorch
  -> record fallback chain
  -> if strict_backend=true: fail instead
```

## 15. Architecture acceptance matrix

| Connection | Required proof |
|---|---|
| CLI -> config | config precedence test and saved effective config |
| config -> recipe | schema validation and recipe hash |
| recipe -> operations | operation factory test |
| operations -> models | mutation plus rollback test |
| models -> layers | model construction and shape contract test |
| layers -> kernels | dispatcher call and backend identity test |
| kernels -> CUDA/Triton/PyTorch | fallback and parity test |
| models -> inference | generation smoke test |
| inference -> integrations | capability and KV-cache translation test |
| runner -> events | event stream test |
| events -> reporting | report contains source run IDs |
| reporting -> results_saver | schema-validated persisted record |
| results_saver -> resume | duplicate/lock/resume test |

## 16. Refactor safety rules

1. Move one folder boundary at a time.
2. Preserve public imports with deprecation shims.
3. Never combine a rename, behavior change, and backend change in one commit.
4. Add an import-boundary test before deleting a compatibility module.
5. Run `rg` for callers before moving a symbol.
6. Keep generated artifacts out of source directories.
7. Do not run bulk formatters over unrelated files.
8. Record every moved file and its replacement in the migration matrix.
9. Test CPU/reference paths before spending GPU time.
10. Treat a process abort or timeout as a failure requiring classification.

## 17. Final folder checklist

```text
[ ] inference has request/result/cache contracts
[ ] integrations expose capability objects
[ ] kernels is a facade, not a second implementation tree
[ ] layers own nn.Module semantics only
[ ] models own composition and checkpoint lifecycle
[ ] operations own model mutations
[ ] ops owns backend implementations and dispatch
[ ] optimization has a defined scope
[ ] pipeline_transform owns transactions and rollback
[ ] recipe owns declarative schemas
[ ] reporting owns evidence and rendering
[ ] results_saver owns persistence
[ ] runners own scheduling and monitoring
[ ] all cross-folder imports follow the dependency graph
[ ] no duplicate writer or dispatcher remains
[ ] no generated files are tracked
[ ] end-to-end recipe-to-report test passes
```

## 18. File-by-file integration manual

This section is the detailed implementation map. The purpose is to remove the
ambiguity of statements such as “connect inference to models”. Each connection
below identifies the producer, consumer, data shape, error behavior, and test.

### 18.1 `neuralops/inference/engine.py`

#### Keep

- request batching;
- prefill/decode scheduling;
- model invocation;
- cache coordination;
- timing boundaries around user-visible generation.

#### Add

```python
from neuralops.runtime.context import RuntimeContext
from neuralops.models.protocols import ModelHandle
from neuralops.inference.contracts import InferenceRequest, InferenceResult


class InferenceEngine:
    def __init__(self, model: ModelHandle, context: RuntimeContext) -> None:
        self.model = model
        self.context = context

    def generate(self, request: InferenceRequest) -> InferenceResult:
        self.context.require_device_available()
        self.context.events.emit("inference.started", request_id=request.request_id)
        # Prefill and decode are delegated to private methods, but all output
        # metrics are assembled into one public result object.
        result = self._generate_impl(request)
        if not torch.isfinite(result.logits).all():
            raise InferenceError("generation produced non-finite logits")
        self.context.events.emit("inference.completed", request_id=request.request_id)
        return result
```

#### Remove or move

- direct imports of vLLM, SGLang, or TensorRT-LLM;
- direct CSV/JSON output;
- hidden `torch.cuda.is_available()` fallback to CPU;
- model surgery calls during generation.

#### Connect to

```text
models.protocols.ModelHandle
runtime.context.RuntimeContext
inference.contracts.InferenceRequest/Result
inference.paged_cache.PagedKVCache
reporting events through RuntimeContext.events
```

#### Test

`tests/architecture/test_inference_engine_contract.py` must prove that the
engine receives a model handle, emits start/end events, and refuses a requested
CUDA device when CUDA is unavailable.

### 18.2 `neuralops/inference/paged_cache.py`

#### Keep

- page allocation;
- append/read operations;
- capacity checks;
- device-local storage.

#### Add

```python
@dataclass(frozen=True)
class KVCacheSpec:
    page_tokens: int
    num_pages: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    device: str
```

The cache must be created from `KVCacheSpec`, not from a vendor-specific config
object. Integrations translate their own cache description into this spec.

#### Required errors

- capacity exceeded -> `CacheCapacityError`;
- shape mismatch -> `CacheShapeError`;
- device mismatch -> `CacheDeviceError`;
- unsupported dtype -> `CacheDtypeError`.

Do not allocate an empty cache as a fallback.

### 18.3 `neuralops/integrations/serving.py`

This is the integration boundary for generic serving, not the inference
implementation itself.

#### Add

```python
class ServingBackend(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResult: ...
    def close(self) -> None: ...


class ServingAdapter(Protocol):
    name: str

    def capabilities(self) -> IntegrationCapabilities: ...
    def create(self, config: ServingConfig) -> ServingBackend: ...
```

#### Exact flow

```text
serving CLI
  -> IntegrationRegistry.resolve(name)
  -> adapter.capabilities()
  -> reject unavailable adapter
  -> adapter.create(config)
  -> InferenceEngine uses ServingBackend protocol
```

The benchmark should record `adapter.name`, adapter version, and capability
state in `environment.json`.

### 18.4 `neuralops/integrations/vllm.py`, `sglang.py`, `tensorrt_llm.py`

Each file should contain only translation code:

```text
NeuralOps InferenceRequest -> vendor request
vendor response -> NeuralOps InferenceResult
NeuralOps KVCacheSpec -> vendor cache configuration
vendor exception -> IntegrationError with original cause
```

Do not put benchmark loops or numerical parity logic in these files. The same
request must be testable with a fake vendor backend without installing the
vendor package.

### 18.5 `neuralops/kernels/extended_kernels.py`

#### Decision

Audit every public symbol in this file:

```bash
rg -n "^(class|def) |__all__" neuralops/kernels/extended_kernels.py
rg -n "extended_kernels" neuralops tests benchmarks scripts
```

For each symbol:

- if it is a public facade, re-export it from `neuralops.kernels`;
- if it contains CUDA/Triton code, move the implementation to `ops/cuda` or
  `ops/triton`;
- if it is a PyTorch oracle, move it to `ops/pytorch`;
- if it is unused, document and remove it in a separate cleanup commit.

The file must not be a fourth backend.

### 18.6 `neuralops/ops/cuda/_bindings.py`

#### Keep

- extension source discovery;
- compiler flags;
- cache naming;
- extension loading.

#### Add

```python
@dataclass(frozen=True)
class CompilationReport:
    name: str
    attempted: bool
    succeeded: bool
    compiler: str | None
    linker: str | None
    error: str | None
    artifact: str | None
```

On failure, return or raise a report that includes the compiler and linker.
Never return an object that looks like a loaded extension when linking failed.

#### Connect to

```text
ops._dispatch -> load_cuda_module
reporting -> CompilationReport
results_saver -> native compilation metadata
```

### 18.7 `neuralops/ops/cuda/_fallback.py`

This is the most important backend boundary.

#### Required algorithm

```python
def dispatch(
    name: str,
    *,
    cuda_fn: Callable[..., Tensor] | None,
    triton_fn: Callable[..., Tensor] | None,
    pytorch_fn: Callable[..., Tensor],
    strict: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> DispatchResult:
    attempts: list[KernelAttempt] = []

    if cuda_fn is not None:
        try:
            output = cuda_fn(*args, **kwargs)
            return DispatchResult(output, "cuda", attempts + [success("cuda")])
        except Exception as exc:
            attempts.append(failure("cuda", exc))
            logger.warning("CUDA kernel %s failed: %s", name, exc)
            if strict:
                raise DispatchError(name, attempts) from exc

    if triton_fn is not None:
        try:
            output = triton_fn(*args, **kwargs)
            return DispatchResult(output, "triton", attempts + [success("triton")])
        except Exception as exc:
            attempts.append(failure("triton", exc))
            logger.warning("Triton kernel %s failed: %s", name, exc)
            if strict:
                raise DispatchError(name, attempts) from exc

    output = pytorch_fn(*args, **kwargs)
    attempts.append(success("pytorch"))
    return DispatchResult(output, "pytorch", attempts)
```

The actual implementation may use different names, but it must preserve this
behavior. `except Exception: pass` is not allowed.

### 18.8 `neuralops/ops/triton/*.py`

Each Triton module must be paired with:

```text
one canonical operator name
one PyTorch reference function
one parity test
one capability declaration
one explicit unsupported-shape error
```

The module must not silently call another Triton kernel and label it with its
own name. If the kernel is not implemented, raise `NotImplementedError` with
the missing operator name.

### 18.9 `neuralops/ops/pytorch/`

This is the mathematical oracle. Each function should be:

- device-agnostic;
- differentiable;
- numerically explicit;
- independent of Triton and CUDA imports;
- covered by shape, causal, finite, and gradient tests.

It must not be considered a performance backend. Its use in a GPU benchmark
must be recorded as `backend_used=pytorch`.

### 18.10 `neuralops/layers/base.py`

Add a common module contract:

```python
class NeuralOpsLayer(nn.Module):
    layer_name: ClassVar[str]
    supports_causal: ClassVar[bool]
    supports_gqa: ClassVar[bool]

    def capability_report(self) -> LayerCapabilities: ...
    def reference_forward(self, *args, **kwargs) -> Tensor: ...
```

Every registered layer should implement or inherit these methods. The registry
uses the capability report for CLI listing and benchmark planning.

### 18.11 Family folders under `neuralops/layers/`

For every family folder, standardize this layout:

```text
family/
  __init__.py        explicit exports only
  <module>.py        nn.Module implementation
  contracts.py       family-specific shape/config contracts if needed
  reference.py       only if reference math is family-specific
  tests/              CPU-safe family tests if the project convention allows
```

Do not put benchmark scripts inside a layer family. Do not import `pytest`,
reporting, or result writers from a layer.

### 18.12 `neuralops/models/configuration.py`

This file should be the bridge between external model config and internal layer
config. Add explicit conversion functions:

```python
def to_layer_config(model_config: PretrainedConfig) -> LayerConfig: ...
def to_runtime_config(model_config: PretrainedConfig) -> RuntimeConfig: ...
```

Do not let each layer read arbitrary Hugging Face config attributes. Normalize
once, validate once, and pass typed config objects downward.

### 18.13 `neuralops/models/hub.py`

Separate source retrieval from model construction:

```text
hub.resolve_source() -> ModelSource
models.loader.load(ModelSource, ModelConfig) -> ModelHandle
```

The hub must not silently download a model when offline mode is requested. It
must report cache miss and the exact remediation.

### 18.14 `neuralops/models/surgical_io.py`

Make checkpoint writes transactional:

```text
destination.tmp/<files>
manifest.json
validation.json
atomic rename destination.tmp -> destination
```

If validation fails, remove only the temporary directory and leave the original
checkpoint untouched. Never create a partial checkpoint with a production file
extension.

### 18.15 `neuralops/operations/attention/replace.py`

Required flow:

```text
resolve target modules
  -> validate target count
  -> resolve canonical attention key
  -> inspect capability report
  -> create replacement modules
  -> copy/initialize weights
  -> run shape and causal checks
  -> return operation result
```

The operation must fail if zero modules match the selector unless the recipe
explicitly permits an empty selection. It must report all replaced module paths.

### 18.16 `neuralops/operations/verify/parity.py`

This should be the shared parity gate for surgery, kernels, and inference. It
must accept reference/candidate tensors, dtype-specific tolerance, and a
context label. It returns a structured report, not a boolean only:

```python
ParityReport(
    passed=False,
    max_diff=0.018,
    mean_diff=0.002,
    tolerance=0.001,
    reference_shape=(2, 4, 64, 32),
    candidate_shape=(2, 4, 64, 32),
    context="gla_scan/cuda",
)
```

### 18.17 `neuralops/pipeline_transform/planner.py`

The planner should be pure: same recipe hash and model metadata produce the
same plan. It must not allocate a CUDA tensor or mutate a model.

The plan must include:

```text
plan_id
recipe_hash
model_source
ordered_operations
required_backends
estimated_memory
rollback_requirements
```

### 18.18 `neuralops/pipeline_transform/executor.py`

The executor should implement:

```text
prepare snapshot
validate all operations
execute operation one
verify operation one
execute operation two
verify operation two
commit all artifacts
```

If any operation fails, restore the latest valid snapshot and mark the entire
run failed unless the recipe explicitly allows partial results.

### 18.19 `neuralops/recipe/loader.py` and `schema.py`

The loader performs only parsing and validation. It should not:

- import arbitrary Python modules;
- download models;
- allocate tensors;
- create directories except an explicitly requested cache location.

Schema validation must reject unknown operation types, unknown backend policy,
negative ranks, invalid tolerances, and missing model source.

### 18.20 `neuralops/reporting/environment.py`

Capture environment once at run start:

```text
python executable/version
torch version
CUDA runtime
GPU name/index/memory
Triton version
NeuralOps git commit/dirty state
selected environment variables without secrets
```

Pass the immutable snapshot to every result record. Do not recapture slightly
different metadata for each module unless the hardware state itself is a
metric.

### 18.21 `neuralops/reporting/evidence.py`

Evidence should reference files and commands, not copy unverified claims:

```python
Evidence(
    command="pytest tests/gpu_exhaustive -v --tb=short",
    host="GPU-3090-1.local",
    exit_code=0,
    log_path="results/run/logs/exhaustive.log",
    summary="915 passed, 2 skipped",
)
```

An evidence record with no exit code or final summary is incomplete.

### 18.22 `neuralops/results_saver/saver.py`

Make the saver accept only validated records:

```python
class ResultsStore:
    def __init__(self, root: Path, policy: StoragePolicy) -> None: ...

    def write(self, record: ResultRecord) -> None:
        record.validate()
        self._write_jsonl(record)
        self._write_json(record)
        if self.policy.parquet_required:
            self._write_parquet(record)
        elif self.policy.parquet_optional:
            self._try_optional_parquet(record)
```

Optional Parquet failure must emit a warning and a metadata flag; it must not
silently change the requested output format.

### 18.23 `neuralops/runners/unified_orchestrator.py`

The orchestrator should be the only component allowed to coordinate multiple
operations. It receives `RunRequest`, creates an execution plan, creates a run
directory, starts workers, collects events, and writes the final summary.

It should not know how a particular attention computes its output. If it has
module-specific `if name == ...` logic, move that logic to the registry or
adapter layer.

### 18.24 `neuralops/runners/monitor/`

The monitor should read `events.jsonl` and result records. It should not import
the worker's private objects. Add a file cursor/checkpoint so the monitor can
resume reading after restart.

## 19. Exact import-boundary rules

These imports should be rejected by a static architecture test:

```text
neuralops.ops -> neuralops.runners
neuralops.ops -> neuralops.reporting
neuralops.layers -> neuralops.runners
neuralops.layers -> neuralops.results_saver
neuralops.results_saver -> neuralops.models
neuralops.reporting -> neuralops.operations (mutation path)
neuralops.integrations -> neuralops.cli
neuralops.recipe -> neuralops.inference
```

These are allowed through protocols or stable contracts:

```text
neuralops.layers -> neuralops.kernels
neuralops.models -> neuralops.layers
neuralops.inference -> neuralops.models
neuralops.inference -> neuralops.integrations
neuralops.runners -> recipe/operations/pipeline/reporting/results_saver
```

Implement the test with AST import inspection rather than importing all modules
on a machine without CUDA dependencies.

## 20. Exact migration sequence for this repository

### Step 1: freeze the public contracts

Add contracts and tests first. Do not move implementation files yet.

### Step 2: map callers

For every file in the folders, produce:

```bash
rg -n "from neuralops\.(inference|integrations|kernels|layers|models|operations|ops|optimization|pipeline_transform|recipe|reporting|results_saver|runners)" neuralops tests benchmarks scripts
```

Record caller, imported symbol, and replacement API.

### Step 3: consolidate dispatch

Move backend selection behind `kernels.dispatch_kernel` and update layers one
family at a time. Keep compatibility imports until tests pass.

### Step 4: consolidate result persistence

Create an adapter from existing benchmark result objects to `ResultRecord`, then
make all new runners use `ResultsStore`.

### Step 5: make operations transactional

Update one operation family at a time: attention replacement, weights,
adaptation, architecture, export.

### Step 6: connect recipe and CLI

CLI creates requests; recipe loader and operation registry do the work. Add
help/examples only after command behavior is tested.

### Step 7: connect inference and integrations

Run smoke tests with a fake backend first, then GPU tests for real integrations.

### Step 8: connect runners and monitoring

Make all runs produce event streams and unique result directories.

### Step 9: remove duplicates

Only now remove duplicate writers, facades, and legacy scripts. Every deletion
must cite its replacement and caller audit.

## 21. Exact files to add for integration coverage

```text
tests/architecture/test_import_boundaries.py
tests/architecture/test_runtime_contracts.py
tests/architecture/test_recipe_to_operation.py
tests/architecture/test_operation_transaction.py
tests/architecture/test_layer_to_kernel_dispatch.py
tests/architecture/test_inference_to_integration.py
tests/architecture/test_reporting_to_results_store.py
tests/architecture/test_runner_resume_and_locking.py
tests/architecture/fixtures/fake_model.py
tests/architecture/fixtures/fake_backend.py
tests/architecture/fixtures/recipes/replace_attention.yaml
```

Every fixture must be deterministic and CPU-safe. GPU tests should be separate
and must not silently downgrade to CPU.

## 22. Exact things to add, remove, and connect

### Add

```text
RuntimeContext
RunRequest / RunResult
RunEvent / EventSink
ModelHandle protocol
KernelRequest / DispatchResult
ServingAdapter protocol
OperationContext / OperationPlan
ResultRecord / ResultsStore
KVCacheSpec
EnvironmentSnapshot
Architecture import tests
Migration matrix
```

### Remove or forbid

```text
duplicate backend dispatchers
duplicate result schemas
result writes from layers/kernels
vendor imports in core modules
arbitrary recipe imports
direct checkpoint mutation from operations without snapshots
shared latest output files
silent fallback after explicit backend request
hard-coded metric defaults
tracked __pycache__ and .DS_Store
```

### Connect

```text
CLI -> RuntimeConfig -> RunRequest
RecipeLoader -> OperationFactory
Operation -> ModelHandle
Layer -> KernelFacade
KernelFacade -> BackendDispatcher
Dispatcher -> DispatchResult -> Reporting
ModelHandle -> InferenceEngine
InferenceEngine -> ServingAdapter
Runner -> EventSink
EventSink -> Reporting
Reporting -> ResultsStore
ResultsStore -> Resume/Compare/Ledger
```

## 23. Final review questions for every folder

Before accepting a folder cleanup, answer all questions:

1. What is the single sentence responsibility of this folder?
2. Which public files are allowed to import it?
3. Which folders is it forbidden to import?
4. What is the input contract?
5. What is the output contract?
6. How does it report failure?
7. How does it preserve provenance?
8. Does it write files? If yes, which writer owns that path?
9. Does it select a backend? If yes, where is the actual backend recorded?
10. Can it run without CUDA for reference tests?
11. What happens when an optional dependency is missing?
12. Which test proves its integration with the next folder?
13. Which files are legacy and why are they retained?
14. What can safely be deleted after migration?

If any answer is unclear, the folder is not yet cleanly integrated.
