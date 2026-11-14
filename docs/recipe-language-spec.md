# NeuralOps Recipe Language Specification v1.0.0

Status: NORMATIVE SPECIFICATION
Version: 1.0.0
API Version: `neuralops/v1`
Date: 2026-10-31
Authors: NeuralOps Core Architecture Team

---

## Section 1 — Introduction and Design Axioms

NeuralOps Recipe Language (NRL) is a declarative, statically typed, versioned domain-specific language for specifying post-training deep learning model transformation pipelines. Rather than relying on imperative Python scripts that mutate state in place, NRL recipes define an explicit, reproducible transformation contract that can be pre-validated, statically checked for structural compatibility, simulated without hardware allocation, safely executed with atomic transactional rollbacks, and certified with cryptographic execution provenance.

The language is governed by four immutable design axioms:

### A1 — Declarative

**The recipe specifies WHAT transformations are to be applied, never HOW they are computed.**
A recipe file is purely data. It contains no imperative control flow (`if`, `when`, `switch`), no iterative loops (`for`, `while`), no variable assignments (`let`, `set`), no computational expressions, and no custom function invocations. If an operator requires different transformation policies depending on model architecture or hardware capacity, multiple discrete recipes must be authored, validated, and selected at the workflow orchestration layer.

*Rationale:* Declarative representations decouple transformation intent from execution engines. A recipe can be inspected, planned, statically analyzed for shape and rank invariants, and validated before touching memory or allocating GPU devices. Imperative logic embedded in transformation files creates opaque side effects, non-reproducible dynamic branches, and unpredictable mutation states.

*Counter-Example Violation:*
```yaml
# VIOLATION OF A1: Contains imperative conditional branch
operations:
  - type: attention.replace
    if: "model.num_heads > 32"  # REJECTED: Imperative control flow
    config:
      target: chunk_gla
```

### A2 — Typed and Versioned

**Every operation, type transition, and recipe schema is explicitly typed and pinned to a semantic version.**
Each operation contract specifies strict input and output state types (e.g., `NeuralOpsModel[loaded]` $\rightarrow$ `NeuralOpsModel[attention_replaced]`). All operation references in recipes must pin a semantic version (`major.minor.patch`). The language specification itself is versioned under `apiVersion: neuralops/v1` and guarantees backward compatibility for all minor and patch iterations.

*Rationale:* Silent runtime behavior drift across library releases is the primary cause of regression in post-training optimization. Pinned operational contracts guarantee that an operation executed six months later evaluates against identical parameter schemas, numerical invariants, and validation gates.

*Counter-Example Violation:*
```yaml
# VIOLATION OF A2: Unversioned type and loose contract
operations:
  - type: attention.replace  # REJECTED: Missing explicit version pinning
    config:
      target: latest         # REJECTED: Floating, unpinned configuration
```

### A3 — Composable with Explicit Dependencies

**Operations form a strictly ordered linear transformation chain with declarative dependency tracking.**
Operations execute in sequential pipeline order ($O_1 \rightarrow O_2 \rightarrow \dots \rightarrow O_n$). The `depends_on` field establishes an explicit declaration of semantic dependency and postcondition consumption, not an automated reordering mechanism. Forward references, circular dependencies, and out-of-order state mutations are statically illegal.

*Rationale:* Linear transformation chains enable deterministic state snapshots and clean backward rollback paths. Allowing arbitrary graph dependencies or auto-reordering introduces combinatorial validation states and makes transactional failure recovery non-deterministic.

*Counter-Example Violation:*
```yaml
# VIOLATION OF A3: Forward dependency and implicit reordering
operations:
  - id: op1
    type: weights.compress
    depends_on: [op2]  # REJECTED: Forward dependency; op2 has not executed
  - id: op2
    type: attention.replace
```

### A4 — Reproducible with Evidence

**A recipe transformation is only certified when accompanied by structured, cryptographic execution evidence.**
The static recipe hash (`recipe_hash` computed over canonicalized bytes) is necessary but insufficient for scientific reproducibility. A complete execution evidence record binds the static recipe hash to the base model git commit SHA, exact framework versions (`torch`, `transformers`, `neuralops`), hardware device metadata, pseudorandom generator seed, per-stage latency/memory telemetry, and validation gate certificates.

*Rationale:* Model surgery modifies billions of numerical parameters. Without verifiable cryptographic evidence binding the inputs, environment, and intermediate verification gates, downstream deployment teams cannot verify whether a deployed checkpoint matches its designated transformation lineage.

*Counter-Example Violation:*
```yaml
# VIOLATION OF A4: Unverifiable external artifact without revision or seed
model:
  source: "meta-llama/Llama-3.1-8B"
  revision: "main"  # REJECTED: Floating branch pointer violates strict reproducibility
execution:
  seed: null        # REJECTED: Unseeded non-deterministic execution
```

---

## Section 2 — Lexical Structure

1. **Surface and Canonical Encodings:**
   The surface representation of a Recipe is valid YAML (1.2). The canonical wire and storage representation is valid JSON (RFC 8259). Both encodings parse into an identical abstract syntax tree (AST).
2. **Value Primitives:**
   The language admits exactly seven primitive types:
   - `string`: UTF-8 encoded text, normalized to Unicode Normalization Form C (NFC) prior to canonicalization.
   - `integer`: 64-bit signed decimal integer (`-9223372036854775808` to `9223372036854775807`).
   - `float`: IEEE 754 double-precision floating-point number. Canonical output renders the shortest equivalent decimal representation (`1.0` is serialized as `1`).
   - `boolean`: Literal `true` or `false`.
   - `null`: Explicit null value (`null` or `~` in YAML, `null` in JSON).
   - `array`: Ordered sequence of zero or more values.
   - `object`: Unordered mapping of unique string keys to values.
3. **Identifiers:**
   All object keys, identifiers, and recipe names must match the DNS-style identifier regular expression:
   `^[a-z0-9][a-z0-9-]{0,62}$`
   Keys must begin with an alphanumeric ASCII character, followed by up to 62 lowercase alphanumeric characters or hyphens.
4. **Operation Type Namespacing:**
   Operation types must conform to dot-delimited hierarchical namespaces:
   `^[a-z]+(\.[a-z0-9_-]+)+$` (e.g., `attention.replace`, `weights.compress`, `export.model`).
5. **Version Strings:**
   All version fields must strictly adhere to Semantic Versioning 2.0.0 (`MAJOR.MINOR.PATCH` with optional `-PRERELEASE` and `+BUILD` metadata). Floating ranges (`^1.0.0`, `~1.2`) are rejected.
6. **Comments:**
   YAML comments (`# ...`) are permitted in surface syntax for human documentation, but are stripped during AST construction and do not survive canonicalization or hashing.

---

## Section 3 — Full EBNF Grammar

The grammar of a NeuralOps Recipe is specified in Extended Backus-Naur Form (EBNF):

```ebnf
Recipe                 ::= "{" WS*
                           '"apiVersion"' WS* ":" WS* ApiVersion WS* "," WS*
                           '"kind"' WS* ":" WS* '"Recipe"' WS* "," WS*
                           '"metadata"' WS* ":" WS* Metadata WS* "," WS*
                           '"model"' WS* ":" WS* ModelSpec WS* "," WS*
                           '"operations"' WS* ":" WS* OperationList WS*
                           ( "," WS* '"validation"' WS* ":" WS* ValidationSpec )?
                           ( "," WS* '"execution"' WS* ":" WS* ExecutionSpec )?
                           ( "," WS* '"artifacts"' WS* ":" WS* ArtifactsSpec )?
                           WS* "}" ;

ApiVersion             ::= '"neuralops/v1"' ;

Metadata               ::= "{" WS*
                           '"name"' WS* ":" WS* Identifier WS* "," WS*
                           '"version"' WS* ":" WS* SemVerString WS* "," WS*
                           '"description"' WS* ":" WS* JSONString
                           ( "," WS* '"author"' WS* ":" WS* JSONString )?
                           ( "," WS* '"created_at"' WS* ":" WS* ISO8601String )?
                           ( "," WS* '"tags"' WS* ":" WS* StringList )?
                           ( "," WS* '"license"' WS* ":" WS* JSONString )?
                           ( "," WS* '"production_ready"' WS* ":" WS* Boolean )?
                           ( "," WS* '"references"' WS* ":" WS* StringList )?
                           WS* "}" ;

ModelSpec              ::= "{" WS*
                           '"source"' WS* ":" WS* JSONString WS* "," WS*
                           '"dtype"' WS* ":" WS* DtypeString
                           ( "," WS* '"revision"' WS* ":" WS* GitCommitHash )?
                           ( "," WS* '"trust_remote_code"' WS* ":" WS* Boolean )?
                           ( "," WS* '"device_map"' WS* ":" WS* JSONString )?
                           ( "," WS* '"family"' WS* ":" WS* Identifier )?
                           WS* "}" ;

OperationList          ::= "[" WS* Operation ( WS* "," WS* Operation )* WS* "]" ;

Operation              ::= "{" WS*
                           ( '"id"' WS* ":" WS* Identifier WS* "," WS* )?
                           '"type"' WS* ":" WS* OperationType WS* "," WS*
                           '"version"' WS* ":" WS* SemVerString WS* "," WS*
                           '"config"' WS* ":" WS* JSONObject
                           ( "," WS* '"depends_on"' WS* ":" WS* IdentifierList )?
                           WS* "}" ;

ValidationSpec         ::= "{" WS*
                           '"profile"' WS* ":" WS* ValidationProfile
                           ( "," WS* '"policy"' WS* ":" WS* ValidationPolicy )?
                           ( "," WS* '"gates"' WS* ":" WS* GateOverrideList )?
                           ( "," WS* '"thresholds"' WS* ":" WS* JSONObject )?
                           WS* "}" ;

ValidationProfile      ::= '"none"' | '"structural"' | '"numerical"' | '"behavioral"' | '"empirical"' | '"production"' ;

ValidationPolicy       ::= '"fail_fast"' | '"continue_on_warning"' | '"collect_all"' ;

ExecutionSpec          ::= "{" WS*
                           ( '"device"' WS* ":" WS* DeviceString WS* ","? )?
                           ( '"seed"' WS* ":" WS* Integer WS* ","? )?
                           ( '"rollback"' WS* ":" WS* Boolean WS* ","? )?
                           ( '"memory_budget_gb"' WS* ":" WS* Number )?
                           WS* "}" ;

ArtifactsSpec          ::= "{" WS*
                           '"output_dir"' WS* ":" WS* JSONString
                           ( "," WS* '"save_model"' WS* ":" WS* Boolean )?
                           ( "," WS* '"save_report"' WS* ":" WS* Boolean )?
                           ( "," WS* '"format"' WS* ":" WS* ExportFormat )?
                           WS* "}" ;

ExportFormat           ::= '"hf"' | '"safetensors"' | '"vllm"' | '"sglang"' | '"onnx"' | '"gguf"' ;

OperationType          ::= '"' [a-z]+ ("." [a-z0-9_-]+)+ '"' ;
Identifier             ::= '"' [a-z0-9] [a-z0-9-]{0,62} '"' ;
SemVerString           ::= '"' [0-9]+ "." [0-9]+ "." [0-9]+ ( "-" [0-9A-Za-z.-]+ )? ( "+" [0-9A-Za-z.-]+ )? '"' ;
DtypeString            ::= '"bfloat16"' | '"float16"' | '"float32"' | '"fp8"' ;
DeviceString           ::= '"auto"' | '"cuda"' | '"cpu"' | '"mps"' ;
GitCommitHash          ::= '"' [0-9a-f]{40} '"' ;
ISO8601String          ::= '"' [0-9]{4} "-" [0-9]{2} "-" [0-9]{2} "T" [0-9]{2} ":" [0-9]{2} ":" [0-9]{2} ( "Z" | [+-] [0-9]{2} ":" [0-9]{2} ) '"' ;
```

---

## Section 4 — Type System

The NRL type system is a static refinement type system over deep learning models and runtime artifacts.

### 4.1 Domain Types

1. **`NeuralOpsModel[state]`**
   Represents an in-memory PyTorch neural network module parameterized by its lifecycle transformation state:
   - `state = loaded`: Model loaded directly from base weights; untouched.
   - `state = attention_replaced`: Quadratic multi-head or grouped-query attention modules transplanted with subquadratic linear/recurrent operators.
   - `state = compressed`: Weight matrices subjected to low-rank SVD, ASVD, or spectral factorizations.
   - `state = quantized`: Weights converted to discrete low-bit representations (FP8, INT4 AWQ/GPTQ).
   - `state = pruned`: Structural decoder layers, attention heads, or MLP intermediate widths removed.
   - `state = adapted`: Post-surgery adapters attached (LoRA) or knowledge-distillation fine-tuning applied.
   - `state = *`: Universal wild-card state matching any lifecycle state.

2. **`ServingArtifact[engine]`**
   Represents a serialized runtime package prepared for production serving engines:
   - `engine ∈ {hf, safetensors, vllm, sglang, tensorrt_llm, tgi, onnx, gguf}`

### 4.2 Operation Type Signatures

| Operation Type | Input Type Signature | Output Type Signature | State Transition Category |
|---|---|---|---|
| `attention.replace` | `NeuralOpsModel[loaded]` | `NeuralOpsModel[attention_replaced]` | Structural Surgery |
| `attention.configure` | `NeuralOpsModel[*]` | `NeuralOpsModel[*]` | Hyperparameter Tuning |
| `weights.compress` | `NeuralOpsModel[*]` | `NeuralOpsModel[compressed]` | Numerical Compression |
| `weights.decompose` | `NeuralOpsModel[*]` | `NeuralOpsModel[compressed]` | Tensor Factorization |
| `weights.quantize` | `NeuralOpsModel[*]` | `NeuralOpsModel[quantized]` | Discrete Quantization |
| `weights.prune` | `NeuralOpsModel[*]` | `NeuralOpsModel[pruned]` | Structural Sparsity |
| `architecture.reduce_mlp` | `NeuralOpsModel[*]` | `NeuralOpsModel[pruned]` | Dimensionality Shrinkage |
| `architecture.drop_layer` | `NeuralOpsModel[*]` | `NeuralOpsModel[pruned]` | Depth Pruning |
| `architecture.modify` | `NeuralOpsModel[*]` | `NeuralOpsModel[*]` | Topology Modification |
| `adaptation.lora` | `NeuralOpsModel[*]` | `NeuralOpsModel[adapted]` | Parameter Efficient Fine-Tuning |
| `adaptation.distill` | `NeuralOpsModel[*]` | `NeuralOpsModel[adapted]` | Recovery Fine-Tuning |
| `adaptation.heal` | `NeuralOpsModel[*]` | `NeuralOpsModel[adapted]` | Micro-Calibration |
| `export.model` | `NeuralOpsModel[*]` | `ServingArtifact[engine]` | Terminal Export |
| `export.checkpoint` | `NeuralOpsModel[*]` | `ServingArtifact[hf]` | Intermediate Checkpoint |
| `verify.parity` | `NeuralOpsModel[S]` | `NeuralOpsModel[S]` | Identity Probe |

### 4.3 Static Verification Capabilities and Preconditions

The type system catches structural order violations at compile time before weights or memory are allocated.
- **Why `export.model` $\rightarrow$ `weights.compress` is caught:**
  `export.model` emits `ServingArtifact[engine]`, which does not satisfy the input constraint `NeuralOpsModel[*]` required by `weights.compress`. The type checker raises `E_COMP_TYPE_MISMATCH`.
- **Why `weights.compress` $\rightarrow$ `attention.replace` requires preconditions:**
  Both operations consume and produce `NeuralOpsModel`. However, replacing an attention layer after compressing weight projections invalidates the low-rank projection factorizations ($U, S, V$). This semantic conflict is caught by static precondition checking (`preconditions: [model.attention_intact == true]`), producing `E_COMP_PRECONDITION_UNSATISFIED`.

---

## Section 5 — Operation Registry Protocol

The operation registry manages the repository of executable transformation contracts.

### 5.1 Registration Contract Interface

Every operation contract exposes the following immutable interface properties:
- `name`: Namespaced identifier (e.g., `attention.replace`).
- `version`: SemVer 2.0.0 string (e.g., `1.0.0`).
- `input_state`: Required `NeuralOpsModel` input state.
- `output_state`: Resulting `NeuralOpsModel` or `ServingArtifact` output state.
- `params_schema`: JSON Schema Draft 7 object defining all configuration parameters.
- `preconditions`: List of static and dynamic predicates evaluated against the model architecture before execution.
- `postconditions`: List of invariant checks evaluated after operation execution.
- `declares_mutations`: Dotted module and parameter path prefixes touched by the operation (used for transactional snapshots).
- `deterministic_given_seed`: Boolean indicating whether identical seeds produce bit-identical weight matrices.
- `estimated_time_and_memory`: Analytic resource estimation function taking model dimensions and producing estimated latency and VRAM footprint.

### 5.2 Coexistence and Versioning Rules

1. Multiple versions of the same operation name coexist within the registry simultaneously (e.g., `weights.compress@1.0.0` and `weights.compress@2.0.0`).
2. Version bumps obey strict rules:
   - **MAJOR:** Any change to input/output state types, removed parameters, breaking changes in numerical behavior, or added required parameters.
   - **MINOR:** Backward-compatible additions of optional parameters or enhanced kernel dispatch options.
   - **PATCH:** Implementation bug fixes preserving exact mathematical and interface contracts.
3. **Deprecation Policy:** Deprecated operation versions emit `E_OP_VERSION_DEPRECATED` warnings during planning. An operation version is never deleted until a major bump of the Recipe Language itself (`apiVersion: neuralops/v2`).

---

## Section 6 — The 15 Initial Operation Contracts

### 6.1 `attention.replace`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[loaded]` $\rightarrow$ `NeuralOpsModel[attention_replaced]`
- **Supported Families:** Standard transformer decoder backbones (`llama`, `gpt2`, `qwen`, `mistral`, `bert`). Excludes recurrent backbones (`mamba`, `rwkv`).
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["target"],
    "properties": {
      "target": {"type": "string", "enum": ["chunk_gla", "retnet", "cosformer", "decay_linear_attention", "hgrn2"]},
      "head_dim": {"type": "integer", "enum": [32, 64, 128]},
      "chunk_size": {"type": "integer", "default": 64},
      "gate_low_rank_dim": {"type": "integer", "default": 16}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `has_standard_attention(model) == true`, `model.config.hidden_size % head_dim == 0`.
- **Postconditions:** `all_attention_modules_match(model, target) == true`.
- **Declares Mutations:** `["*.self_attn", "*.attn"]`.
- **Deterministic Given Seed:** `true`.

### 6.2 `attention.configure`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[*]`
- **Supported Families:** All transformer families.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "chunk_size": {"type": "integer", "minimum": 16},
      "clamp_min": {"type": "number"},
      "clamp_max": {"type": "number"}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `model.has_attention == true`.
- **Postconditions:** `attention_parameters_configured(model) == true`.
- **Declares Mutations:** `["*.attn.config"]`.
- **Deterministic Given Seed:** `true`.

### 6.3 `weights.compress`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[compressed]`
- **Supported Families:** All dense and transformer architectures.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["method", "target_ratio"],
    "properties": {
      "method": {"type": "string", "enum": ["svd", "asvd", "fwsvd", "hnc_svd"]},
      "target_ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95},
      "rank_step": {"type": "integer", "default": 32},
      "whitening": {"type": "boolean", "default": false}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `uncompressed_linear_weights_exist(model) == true`.
- **Postconditions:** `model.compression_ratio <= target_ratio`.
- **Declares Mutations:** `["*.mlp.*", "*.self_attn.*_proj"]`.
- **Deterministic Given Seed:** `true`.

### 6.4 `weights.decompose`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[compressed]`
- **Supported Families:** All models containing dense linear layers.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["rank"],
    "properties": {
      "rank": {"type": "integer", "minimum": 4},
      "targets": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `linear_targets_exist(model, targets) == true`.
- **Postconditions:** `all_targets_decomposed_to_rank(model, rank) == true`.
- **Declares Mutations:** `["*"]`.
- **Deterministic Given Seed:** `true`.

### 6.5 `weights.quantize`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[quantized]`
- **Supported Families:** `llama`, `gpt2`, `qwen`, `mistral`.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["bits", "method"],
    "properties": {
      "bits": {"type": "integer", "enum": [4, 8]},
      "method": {"type": "string", "enum": ["fp8", "int4_awq", "int4_gptq"]},
      "group_size": {"type": "integer", "default": 128}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `torch.cuda.is_available() == true`, `calibration_data_available(method) == true`.
- **Postconditions:** `all_linear_weights_quantized(model, bits) == true`.
- **Declares Mutations:** `["*.weight"]`.
- **Deterministic Given Seed:** `false`.

### 6.6 `weights.prune`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[pruned]`
- **Supported Families:** All dense architectures.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["sparsity", "pattern"],
    "properties": {
      "sparsity": {"type": "number", "minimum": 0.1, "maximum": 0.9},
      "pattern": {"type": "string", "enum": ["unstructured", "2:4", "channel"]}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `has_dense_parameters(model) == true`.
- **Postconditions:** `measured_sparsity(model) >= sparsity`.
- **Declares Mutations:** `["*.weight"]`.
- **Deterministic Given Seed:** `true`.

### 6.7 `architecture.reduce_mlp`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[pruned]`
- **Supported Families:** Transformer decoders with SwiGLU or GELU MLPs (`llama`, `gpt2`, `qwen`).
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["ratio"],
    "properties": {
      "ratio": {"type": "number", "minimum": 0.2, "maximum": 0.95}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `has_standard_mlp_blocks(model) == true`.
- **Postconditions:** `mlp_intermediate_dim(model) == original_dim * ratio`.
- **Declares Mutations:** `["*.mlp", "*.feed_forward"]`.
- **Deterministic Given Seed:** `true`.

### 6.8 `architecture.drop_layer`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[pruned]`
- **Supported Families:** Multi-layer transformer decoders and encoders.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["drop_count"],
    "properties": {
      "drop_count": {"type": "integer", "minimum": 1},
      "strategy": {"type": "string", "enum": ["tail", "interleaved", "uniform"], "default": "tail"}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `num_layers(model) > drop_count`.
- **Postconditions:** `num_layers(model) == original_num_layers - drop_count`.
- **Declares Mutations:** `["model.layers", "transformer.h", "encoder.layer"]`.
- **Deterministic Given Seed:** `true`.

### 6.9 `architecture.modify`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[*]`
- **Supported Families:** All.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["transformation_key"],
    "properties": {
      "transformation_key": {"type": "string"},
      "parameters": {"type": "object"}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `transformation_key_supported(transformation_key) == true`.
- **Postconditions:** `transformation_applied(transformation_key) == true`.
- **Declares Mutations:** `["*"]`.
- **Deterministic Given Seed:** `true`.

### 6.10 `adaptation.lora`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[adapted]`
- **Supported Families:** All models containing `nn.Linear`.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["rank", "alpha"],
    "properties": {
      "rank": {"type": "integer", "minimum": 1},
      "alpha": {"type": "number", "minimum": 1.0},
      "targets": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `model_weights_frozen_or_ready(model) == true`.
- **Postconditions:** `lora_adapters_attached(model) == true`.
- **Declares Mutations:** `["*.lora_A", "*.lora_B"]`.
- **Deterministic Given Seed:** `true`.

### 6.11 `adaptation.distill`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[adapted]`
- **Supported Families:** Transformer backbones with matching student/teacher tokenizers.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["teacher_source", "steps"],
    "properties": {
      "teacher_source": {"type": "string"},
      "steps": {"type": "integer", "minimum": 10},
      "learning_rate": {"type": "number", "default": 5e-5},
      "pillars": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `torch.cuda.is_available() == true`, `dataset_specified() == true`.
- **Postconditions:** `recovery_loss_converged(model) == true`.
- **Declares Mutations:** `["*"]`.
- **Deterministic Given Seed:** `true`.

### 6.12 `adaptation.heal`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `NeuralOpsModel[adapted]`
- **Supported Families:** Transformer backbones after attention transplant or layer pruning.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["calibration_tokens"],
    "properties": {
      "calibration_tokens": {"type": "integer", "minimum": 1024},
      "mode": {"type": "string", "enum": ["norm_only", "bias_only", "full_quick"]}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `model.is_mutated == true`.
- **Postconditions:** `model.is_calibrated == true`.
- **Declares Mutations:** `["*.ln_*", "*.norm*"]`.
- **Deterministic Given Seed:** `true`.

### 6.13 `export.model`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `ServingArtifact[engine]`
- **Supported Families:** All models.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["format"],
    "properties": {
      "format": {"type": "string", "enum": ["hf", "safetensors", "vllm", "sglang", "onnx", "gguf"]},
      "verify_reload": {"type": "boolean", "default": true}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `is_finite_model(model) == true`.
- **Postconditions:** `destination_directory_contains_valid_artifact(output_dir) == true`.
- **Declares Mutations:** `[]`.
- **Deterministic Given Seed:** `true`.

### 6.14 `export.checkpoint`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[*]` $\rightarrow$ `ServingArtifact[hf]`
- **Supported Families:** All models.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "save_tokenizer": {"type": "boolean", "default": true}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `valid_model_state(model) == true`.
- **Postconditions:** `checkpoint_file_exists() == true`.
- **Declares Mutations:** `[]`.
- **Deterministic Given Seed:** `true`.

### 6.15 `verify.parity`
- **Version:** `1.0.0`
- **Input / Output State:** `NeuralOpsModel[S]` $\rightarrow$ `NeuralOpsModel[S]`
- **Supported Families:** All models.
- **Params Schema:**
  ```json
  {
    "type": "object",
    "required": ["max_abs_diff"],
    "properties": {
      "max_abs_diff": {"type": "number", "minimum": 0.0},
      "cosine_threshold": {"type": "number", "minimum": -1.0, "maximum": 1.0}
    },
    "additionalProperties": false
  }
  ```
- **Preconditions:** `baseline_output_tensor_cached() == true`.
- **Postconditions:** `parity_check_passed() == true`.
- **Declares Mutations:** `[]`.
- **Deterministic Given Seed:** `true`.

---

## Section 7 — Well-Formedness Rules (W1–W15)

A recipe document must satisfy all fifteen static well-formedness rules:

- **W1 — Root Schema Conformance:** The document must match top-level fields: `apiVersion`, `kind`, `metadata`, `model`, `operations`.
  *Error:* `E_SCHEMA_MISSING_FIELD`
  *Violation:* `kind: "Recipe"` omitted.
- **W2 — SemVer Validity:** Version strings in `metadata.version` and `operations[i].version` must parse as valid SemVer 2.0.0.
  *Error:* `E_SCHEMA_INVALID_SEMVER`
  *Violation:* `version: "v1.2"`.
- **W3 — Operation Registry Membership:** Every operation `type` must exist in the active operation registry.
  *Error:* `E_REGISTRY_UNKNOWN_OP`
  *Violation:* `type: "quantum.teleport"`.
- **W4 — Operation Version Compatibility:** The pinned version in `operations[i].version` must exist in the registry for that operation name.
  *Error:* `E_REGISTRY_UNKNOWN_VERSION`
  *Violation:* `type: "attention.replace", version: "99.0.0"`.
- **W5 — Strict Parameter Schema Conformance:** The `config` object must validate against the operation's JSON Schema with `additionalProperties: false`.
  *Error:* `E_SCHEMA_UNKNOWN_FIELD`
  *Violation:* `config: { target: "chunk_gla", bogus_option: 42 }`.
- **W6 — Fail-Closed Schema Validation:** Unknown fields at any level of the recipe document trigger immediate rejection.
  *Error:* `E_SCHEMA_UNKNOWN_FIELD`
  *Violation:* `metadata: { name: "test", experimental_flag: true }`.
- **W7 — Unique Operation Identifiers:** Every explicit operation `id` must be unique across the recipe.
  *Error:* `E_COMP_DUPLICATE_ID`
  *Violation:* Two operations sharing `id: "step-1"`.
- **W8 — Declared Dependency Resolution:** Every element in `depends_on` must reference a valid preceding operation `id`.
  *Error:* `E_COMP_FORWARD_DEPENDENCY`
  *Violation:* `depends_on: ["non_existent_op"]`.
- **W9 — Strictly Preceding Dependencies:** No operation may depend on itself or an operation appearing later in the array.
  *Error:* `E_COMP_FORWARD_DEPENDENCY`
  *Violation:* Operation at index 0 declaring `depends_on: ["op-2"]`.
- **W10 — Linear Type Chain Validity:** The output type of operation $i$ must unify with the input type of operation $i+1$.
  *Error:* `E_COMP_TYPE_MISMATCH`
  *Violation:* `export.model` (`ServingArtifact`) followed by `weights.compress` (`NeuralOpsModel`).
- **W11 — Precondition Satisfiability on Declared Family:** Declared model family must satisfy all static preconditions of each operation.
  *Error:* `E_COMP_PRECONDITION_UNSATISFIED`
  *Violation:* `attention.replace` declared on `model: { family: "mamba" }`.
- **W12 — Disallow Incompatible Self-Composition:** Multiple applications of identical operations must be explicitly permitted by the operation contract.
  *Error:* `E_COMP_SELF_COMPOSE_DISALLOWED`
  *Violation:* Applying `attention.replace` twice in the same pipeline.
- **W13 — Empirical Profile Requires Evaluation Dataset:** If `validation.profile` is set to `empirical` or `production`, an evaluation dataset must be provided or policy configured.
  *Error:* `E_VALIDATION_NO_DATASET`
  *Violation:* Profile is `empirical` but no dataset or calibration corpus is configured.
- **W14 — Valid Gate Threshold References:** Threshold configuration overrides must reference gates registered in the validation engine.
  *Error:* `E_VALIDATION_UNKNOWN_GATE`
  *Violation:* `thresholds: { imaginary_loss_gate: 0.1 }`.
- **W15 — Production Safety Gate Contract:** If `validation.profile` is `"none"`, the recipe must explicitly declare `metadata.production_ready: false`.
  *Error:* `E_VALIDATION_LEVEL_INVALID`
  *Violation:* Profile `"none"` with `production_ready: true` or omitted.

---

## Section 8 — Operational Semantics (E1–E6)

The recipe executor operates as a deterministic, transactional state machine:

- **E1 — Execution Loop:**
  The engine executes operations sequentially:
  `For op in recipe.operations:`
  1. Resolve operational contract from registry.
  2. Verify all `depends_on` operations succeeded.
  3. Evaluate operation `preconditions` against model.
  4. Capture atomic `snapshot` over `op.declares_mutations`.
  5. Apply operation: `(model_next, op_evidence) = op.apply(model_curr, op.config)`.
  6. Execute `run_validation_gates` at configured profile.
  7. If any required gate fails: invoke `rollback`, abort, and raise `E_EXEC_VALIDATION_FAILED`.
  8. If gates pass: commit stage, append evidence record to chain, release snapshot.

- **E2 — Snapshot Scope:**
  Snapshots capture only state paths listed in `op.declares_mutations`. The snapshot stores parameter tensors, buffers, sub-module references, model config attributes, and pseudorandom generator states. Unmutated sub-modules are never cloned.

- **E3 — Transactional Rollback:**
  Upon gate failure or unhandled exception during operation execution, `restore(model, snapshot)` restores the mutated state paths bit-for-bit. If the rollback restore fails, the engine raises `E_EXEC_ROLLBACK_FAILED` and halts immediately.

- **E4 — Validation Shortcut Restriction:**
  When `profile == "none"`, validation gates are bypassed for rapid testing. This execution path is rejected unless `metadata.production_ready: false` is explicitly set.

- **E5 — Empirical Fallback:**
  When `profile == "empirical"` and physical hardware or dataset is unavailable, empirical gates return `skipped: true` (never `passed: true`). The validation status reflects `PARTIAL` rather than `SUCCESS`.

- **E6 — Dry-Run Planning:**
  The planning phase (`plan_recipe`) executes all W1–W15 checks, verifies type transitions, evaluates preconditions against model configuration metadata, and estimates VRAM and latency without loading weight tensors or creating CUDA contexts.

---

## Section 9 — Composition Rules (C1–C7)

- **C1 — Linear State Threading:** The output model state of stage $i$ forms the input model state of stage $i+1$.
- **C2 — Declarative Intent:** The `depends_on` attribute declares logical dependency; it does not alter execution sequence.
- **C3 — Multiple Predecessors:** An operation may depend on multiple preceding stages (`depends_on: [op1, op2]`); all named stages must complete successfully.
- **C4 — No Forward References:** An operation cannot reference a stage that appears later in the document.
- **C5 — Acyclicity:** Strict compliance with C4 guarantees that the dependency graph is a directed acyclic graph (DAG).
- **C6 — Cascade Skip Semantics:** If an optional stage is skipped, all downstream operations that depend upon it must also be skipped, or the pipeline is invalid.
- **C7 — Verifiable Evidence Chain:** Each stage appends a signed evidence record containing `input_state_hash`, `output_state_hash`, gate certificates, duration, and peak memory allocated.

---

## Section 10 — Versioning Policy

NeuralOps maintains three distinct, independent versioning streams:

1. **Language Version (`apiVersion: neuralops/v1`):**
   Governs the grammar, top-level keys, well-formedness rules (W1–W15), and canonical hashing algorithm. A change to language semantics or canonical hashing requires `apiVersion: neuralops/v2`.
2. **Operation Contract Version (`type: name, version: X.Y.Z`):**
   Governed by individual operation definitions. A breaking parameter change in `weights.compress` increments its major version (`2.0.0`) without bumping the language version.
3. **Library Package Version (`neuralops.__version__`):**
   The runtime software release (e.g., `2.1.0`), independent of recipe contracts. A recipe does not pin the library package version.

---

## Section 11 — Canonical Form and Hashing

To compute a deterministic cryptographic hash (`recipe_hash`):

1. **Stripping:** Comments and YAML formatting are removed.
2. **Key Ordering:** All dictionary keys are sorted lexicographically by UTF-8 byte value at all nesting depths.
3. **Whitespace Normalization:** Insignificant whitespace is eliminated (colon followed by single space, comma followed by single space).
4. **Number Formatting:** Numbers use the shortest equivalent representation (`1` instead of `1.0` if integer).
5. **String Encoding:** Strings are normalized using Unicode Normalization Form C (NFC) and UTF-8 encoded.
6. **Hash Computation:** The canonical UTF-8 byte string is digested with SHA-256. The resulting string is prefixed as `sha256:<64_hex_digits>`.

Two recipes are semantically identical if and only if their computed `recipe_hash` values are equal.

---

## Section 12 — Error Taxonomy

| Category | Error Code | Description | Example Trigger | Recommended Remediation |
|---|---|---|---|---|
| **Parse** | `E_PARSE_JSON_SYNTAX` | Invalid JSON syntax encountered. | Trailing comma in JSON payload. | Validate JSON with standard parser. |
| **Parse** | `E_PARSE_YAML_SYNTAX` | Malformed YAML syntax. | Indentation mismatch on operation list. | Align indentation to 2 spaces. |
| **Parse** | `E_PARSE_DUP_KEY` | Duplicate key in single mapping. | Two `version:` keys in metadata. | Remove duplicate key definition. |
| **Schema** | `E_SCHEMA_MISSING_FIELD` | Required field is missing. | Missing `metadata.name`. | Provide required field in recipe. |
| **Schema** | `E_SCHEMA_UNKNOWN_FIELD` | Unknown property rejected (fail-closed). | `metadata.extra_key: 123`. | Remove undeclared field. |
| **Schema** | `E_SCHEMA_WRONG_TYPE` | Field value type mismatch. | `seed: "123"` (string instead of int). | Pass correct primitive type. |
| **Schema** | `E_SCHEMA_INVALID_SEMVER` | Version string violates SemVer 2.0.0. | `version: "v1.0"`. | Use strict `X.Y.Z` format. |
| **Schema** | `E_SCHEMA_INVALID_IDENTIFIER` | Key or identifier violates regex. | `name: "My Recipe!"`. | Use lowercase alphanumeric and hyphens. |
| **Registry** | `E_REGISTRY_UNKNOWN_OP` | Operation type not registered. | `type: "matrix.quantum_solve"`. | Check registered operation names. |
| **Registry** | `E_REGISTRY_UNKNOWN_VERSION` | Operation version not found. | `type: "attention.replace", version: "9.9.9"`. | Check supported operation versions. |
| **Registry** | `E_REGISTRY_DUPLICATE` | Duplicate registration attempt. | Registering same op and version twice. | Increment version or use override. |
| **Composition** | `E_COMP_DUPLICATE_ID` | Non-unique operation `id`. | Two operations with `id: "compress"`. | Assign unique identifiers. |
| **Composition** | `E_COMP_FORWARD_DEPENDENCY` | Dependency references future stage. | Step 1 depends on Step 2. | Reorder pipeline or fix `depends_on`. |
| **Composition** | `E_COMP_TYPE_MISMATCH` | Incompatible lifecycle states. | `export.model` before `weights.compress`. | Place export at pipeline termination. |
| **Composition** | `E_COMP_PRECONDITION_UNSATISFIED` | Precondition check failed. | `attention.replace` on Mamba model. | Ensure target model supports operation. |
| **Composition** | `E_COMP_SELF_COMPOSE_DISALLOWED` | Operation applied multiple times. | Running attention replace twice. | Remove redundant operation. |
| **Validation** | `E_VALIDATION_NO_DATASET` | Empirical profile missing corpus. | Profile `empirical` without dataset. | Supply dataset or lower profile. |
| **Validation** | `E_VALIDATION_UNKNOWN_GATE` | Unknown gate referenced in config. | Override on `non_existent_gate`. | Verify gate identifier spelling. |
| **Validation** | `E_VALIDATION_LEVEL_INVALID` | Profile `none` on production recipe. | Profile `none` with `production_ready: true`. | Set `production_ready: false` or validate. |
| **Execution** | `E_EXEC_OPERATION_FAILED` | Internal error applying operation. | Out-of-memory or kernel crash. | Inspect operation configuration. |
| **Execution** | `E_EXEC_VALIDATION_FAILED` | Required validation gate failed. | Cosine similarity below threshold. | Adjust thresholds or add recovery stage. |
| **Execution** | `E_EXEC_ROLLBACK_FAILED` | Failed to restore pre-stage snapshot. | State dict mismatch during restore. | Fatal error: reload base model. |
| **Execution** | `E_EXEC_SNAPSHOT_FAILED` | Memory error taking snapshot. | Insufficient CPU RAM for state paths. | Reduce snapshot scope or memory budget. |
| **Execution** | `E_EXEC_BASE_MODEL_LOAD_FAILED` | Base model checkpoint missing/corrupted. | Bad HuggingFace repo path. | Verify model path and revision SHA. |
| **Version** | `E_LANG_VERSION_TOO_NEW` | Recipe apiVersion unsupported. | `apiVersion: "neuralops/v9"`. | Upgrade neuralops package. |
| **Version** | `E_LANG_VERSION_TOO_OLD` | Deprecated apiVersion. | `apiVersion: "neuralops/v0"`. | Upgrade recipe to `neuralops/v1`. |
| **Version** | `E_OP_VERSION_DEPRECATED` | Deprecated operation version used. | Operation marked for sunset. | Update operation version in recipe. |

---

## Section 13 — Extension Protocol

Third-party packages may register custom operations using the extension protocol:

```python
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

class CustomAttentionSurgery(Operation):
    contract = OperationContract(
        name="custom.sparse_attention",
        version="1.0.0",
        input_state="loaded",
        output_state="attention_replaced",
        params_schema={
            "type": "object",
            "required": ["sparsity_pattern"],
            "properties": {
                "sparsity_pattern": {"type": "string"}
            },
            "additionalProperties": False,
        },
        preconditions=["has_standard_attention"],
        postconditions=["verify_sparse_pattern"],
        declares_mutations=["*.self_attn"],
        deterministic_given_seed=True,
    )

    def apply(self, model, config):
        # Implementation logic mutating declared paths
        return transformed_model, {"sparsity_applied": config["sparsity_pattern"]}

register_operation(CustomAttentionSurgery())
```

Namespace Rules:
- Custom operations must use namespaced identifiers with at least one dot (`company.operation`).
- Registration fails if `(name, version)` collides with an existing registration.

---

## Section 14 — Conformance Test Suite Specification

This section defines 30+ test recipes with expected outcomes for verification:

### Category 1: Valid Recipes (5 Cases)
1. **Case 1.1 — Minimal Valid Recipe:**
   Single `attention.replace` operation with required fields. $\rightarrow$ `VALID`
2. **Case 1.2 — Multi-Stage Valid Pipeline:**
   `attention.replace` $\rightarrow$ `weights.compress` $\rightarrow$ `export.model`. $\rightarrow$ `VALID`
3. **Case 1.3 — Pruning and Healing Pipeline:**
   `architecture.drop_layer` $\rightarrow$ `adaptation.heal` $\rightarrow$ `export.checkpoint`. $\rightarrow$ `VALID`
4. **Case 1.4 — Full Recipe with Explicit Metadata and Overrides:**
   All optional metadata fields, `validation` thresholds, and `execution` seed provided. $\rightarrow$ `VALID`
5. **Case 1.5 — Micro-Quantization Pipeline:**
   `weights.quantize` with method `fp8` and `export.model`. $\rightarrow$ `VALID`

### Category 2: Schema Violations (5 Cases)
6. **Case 2.1 — Missing `apiVersion`:** $\rightarrow$ `E_SCHEMA_MISSING_FIELD`
7. **Case 2.2 — Unknown Field in Metadata:** $\rightarrow$ `E_SCHEMA_UNKNOWN_FIELD`
8. **Case 2.3 — Non-SemVer Version String:** `version: "1.0"` $\rightarrow$ `E_SCHEMA_INVALID_SEMVER`
9. **Case 2.4 — Illegal Identifier Characters:** `id: "op#1"` $\rightarrow$ `E_SCHEMA_INVALID_IDENTIFIER`
10. **Case 2.5 — Wrong Type for Seed:** `seed: "123"` $\rightarrow$ `E_SCHEMA_WRONG_TYPE`

### Category 3: Registry Violations (5 Cases)
11. **Case 3.1 — Unregistered Operation:** `type: "quantum.entropy"` $\rightarrow$ `E_REGISTRY_UNKNOWN_OP`
12. **Case 3.2 — Unregistered Version:** `type: "attention.replace", version: "9.9.9"` $\rightarrow$ `E_REGISTRY_UNKNOWN_VERSION`
13. **Case 3.3 — Missing Required Parameter in Config:** `attention.replace` without `target` $\rightarrow$ `E_SCHEMA_MISSING_FIELD`
14. **Case 3.4 — Undeclared Parameter in Config:** `target_ratio` passed to `attention.replace` $\rightarrow$ `E_SCHEMA_UNKNOWN_FIELD`
15. **Case 3.5 — Parameter Out of Range:** `target_ratio: 1.5` for `weights.compress` $\rightarrow$ `E_SCHEMA_WRONG_TYPE`

### Category 4: Composition Violations (5 Cases)
16. **Case 4.1 — Duplicate Operation Identifiers:** Two stages with `id: "stage-1"` $\rightarrow$ `E_COMP_DUPLICATE_ID`
17. **Case 4.2 — Forward Reference Dependency:** Step 1 depends on Step 2 $\rightarrow$ `E_COMP_FORWARD_DEPENDENCY`
18. **Case 4.3 — Invalid Lifecycle Type Transition:** `export.model` followed by `weights.compress` $\rightarrow$ `E_COMP_TYPE_MISMATCH`
19. **Case 4.4 — Architecture Incompatibility:** `attention.replace` on Mamba $\rightarrow$ `E_COMP_PRECONDITION_UNSATISFIED`
20. **Case 4.5 — Disallowed Self-Composition:** Consecutive `attention.replace` $\rightarrow$ `E_COMP_SELF_COMPOSE_DISALLOWED`

### Category 5: Validation Configuration Violations (5 Cases)
21. **Case 5.1 — Empirical Profile Without Dataset:** Profile `empirical` without dataset $\rightarrow$ `E_VALIDATION_NO_DATASET`
22. **Case 5.2 — Unknown Gate in Thresholds:** `thresholds: { fake_gate: 0.5 }` $\rightarrow$ `E_VALIDATION_UNKNOWN_GATE`
23. **Case 5.3 — None Profile with Production Flag:** Profile `none` with `production_ready: true` $\rightarrow$ `E_VALIDATION_LEVEL_INVALID`
24. **Case 5.4 — Invalid Validation Profile Name:** Profile `"ultra"` $\rightarrow$ `E_SCHEMA_WRONG_TYPE`
25. **Case 5.5 — Negative Threshold Value for Absolute Metric:** `max_abs_diff: -0.1` $\rightarrow$ `E_SCHEMA_WRONG_TYPE`

### Category 6: Canonical Form and Hashing Tests (5 Cases)
26. **Case 6.1 — Whitespace Invariance:** Space/newline variants produce identical SHA-256 hash.
27. **Case 6.2 — Key Reordering Invariance:** Permuted mapping keys produce identical SHA-256 hash.
28. **Case 6.3 — Description Sensitivity:** Mutating description changes SHA-256 hash.
29. **Case 6.4 — Operation Order Sensitivity:** Permuting operations changes SHA-256 hash.
30. **Case 6.5 — Pinned Version Sensitivity:** Changing operation version changes SHA-256 hash.

---

## Section 15 — Non-Goals and Explicit Exclusions

The following capabilities are deliberately excluded from NeuralOps Recipe Language v1.0.0:

1. **No Conditional Logic (`if`, `when`, `ternary`):** Prevents non-deterministic execution branches and preserves static plan verification.
2. **No Iterative Loops (`for`, `while`, `repeat`):** Transformation length must be statically bounded and unrollable.
3. **No Variables or Macros (`let`, `const`, `eval`):** Eliminates scoping ambiguity and variable capture issues.
4. **No Custom Functions or Lambdas:** Prevents execution of arbitrary code within recipe definitions.
5. **No User-Defined Types:** Type system is closed and statically known to the compiler.
6. **No Recursion:** All transformations are strictly finite linear sequences.
7. **No Embedded Foreign Scripts (No Python, Bash, Lua):** Eliminates sandbox escape vectors and guarantees pure declarative data.
8. **No Parallel Execution Branches (DAG):** v1 is strictly linear ($O_1 \rightarrow O_2 \dots$); DAG scheduling is reserved for v2.0.
9. **No Inline Numerical Computation:** All numeric values must be explicit literals.
10. **No Side Effects Outside Model State:** No network requests (except via terminal export adapters), no file system writes outside `artifacts.output_dir`, and no untracked telemetry.
