# Architecture Decision Records (ADRs)

## ADR-001: Tuple Return (output, attention_weights, past_kv)
### Context
HuggingFace decoder transformer blocks unpack a 3-element tuple for hidden states, attention maps, and key-value cache states.

### Decision
Return `(output, attention_weights, past_kv)` across all forward calls.

### Rationale
Ensures seamless integration into HF Transformers `modeling_*.py` blocks without specialized output adapters.

### Consequences
**Easier:**
- Direct drop-in compatibility with HF models.
- Consistent interface for caching and downstream generation loops.
**Harder:**
- Callers requiring only tensors must unpack `out[0]`.

### Alternatives Considered
- **Alt 1: Return Tensor only** — Rejected because it breaks standard HuggingFace decoder layers.
- **Alt 2: Custom dataclass** — Rejected due to runtime overhead and impedance mismatch with legacy PyTorch modules.
- **Alt 3: Return dict** — Rejected because standard HuggingFace attention layers do not use dictionaries.

---

## ADR-002: Weight Copy Error Raising
### Context
Weight transfer errors during model surgery (e.g., mismatched shapes) can lead to silent random initialization.

### Decision
Raise explicit `WeightCopyError` on missing weights or incompatible tensor geometries.

### Rationale
Early, fail-fast exception prevents expensive training runs on corrupted models.

### Consequences
**Easier:**
- Instant detection of surgery mistakes.
- Clean debugging traceback.
**Harder:**
- Callers must ensure matching layer configurations before calling `copy_weights`.

### Alternatives Considered
- **Alt 1: Silent warning and skip** — Rejected because silent random init destroys model weights unnoticed.
- **Alt 2: Auto-slice tensors** — Rejected because slicing changes semantic parameter mapping.
- **Alt 3: Return boolean success flag** — Rejected because callers frequently ignore return codes.

---

## ADR-003: Centralized Backend Dispatch
### Context
Target environments vary between Triton GPU, CUDA native, and CPU.

### Decision
Centralize kernel routing in `neuralops.core.backend.dispatch()`.

### Rationale
Consolidates hardware selection logic into a single decoupled subsystem.

### Consequences
**Easier:**
- Clean attention layer forward logic without nested device if/else branches.
- Central place to optimize hardware backends.
**Harder:**
- Slightly higher function call indirection.

### Alternatives Considered
- **Alt 1: In-place `if x.is_cuda:` in layer** — Rejected because it leads to duplicated code across attention layers.
- **Alt 2: C++ extension library** — Rejected due to complex compilation requirements across host machines.
- **Alt 3: Dynamic module replacement** — Rejected because monkey-patching obscures call graphs.

---

## ADR-004: Pure Linear Projections for GPT-2 Surgery
### Context
Pretrained GPT-2 attention projections were trained on softmax attention statistics. Inserting RMSNorm before `out_proj` alters variance profiles.

### Decision
Remove intermediate RMSNorm between linear attention state reduction and `out_proj`.

### Rationale
Maintains matching statistical distribution for pretrained weights and prevents logit divergence.

### Consequences
**Easier:**
- Logit output variance matches pretrained distribution.
- Zero-shot and pre-KD output stability.
**Harder:**
- Long sequence gradient stability requires log-space gating.

### Alternatives Considered
- **Alt 1: Keep RMSNorm with learnable gain** — Rejected because initialization shift still explodes early logits.
- **Alt 2: LayerNorm instead of RMSNorm** — Rejected because mean-centering adds memory without solving the variance shift.
- **Alt 3: Post-hoc projection rescaling** — Rejected because scale factors vary across layers and sequence lengths.

---

## ADR-005: Auxiliary Loss Weight Scheduling
### Context
Attention layer output alignment is essential during early stage KD, but rigid intermediate alignment can hinder end-to-end task performance in later stages.

### Decision
Schedule auxiliary distillation loss weight from `max_weight` to `final_weight` via `LayerDistillScheduler`.

### Rationale
Guides initial convergence while allowing model representation flexibility during fine-tuning.

### Consequences
**Easier:**
- Fast early alignment with lower final perplexity.
**Harder:**
- Introduces scheduler hyperparameter tuning.

### Alternatives Considered
- **Alt 1: Constant auxiliary weight** — Rejected because it overly constrains late-stage representations.
- **Alt 2: Step-function drop** — Rejected because sudden loss landscape shifts induce gradient shocks.
- **Alt 3: Exponential decay** — Rejected because linear decay is simpler and more predictable.

---

## ADR-006: Silent Backend Fallback
### Context
Environments without Triton or CUDA should run verification tests and CPU inference without crashes.

### Decision
Silently fall back to PyTorch implementation when specialized hardware or kernels are unavailable.

### Rationale
Provides universal portability across developer laptops, CI pipelines, and production servers.

### Consequences
**Easier:**
- Tests run cleanly anywhere.
**Harder:**
- Users may unknowingly run slower CPU paths unless logging is checked.

### Alternatives Considered
- **Alt 1: Hard crash with ImportError** — Rejected because it breaks CI and non-GPU environments.
- **Alt 2: Interactive confirmation** — Rejected because autonomous training loops cannot prompt users.
- **Alt 3: Warning log on every call** — Rejected because it clutters logs during high-throughput execution.

---

## ADR-007: HuggingFace Registration via AutoModel.register
### Context
HuggingFace Transformers ecosystem uses auto-classes (`AutoConfig`, `AutoModelForCausalLM`) for model instantiation.

### Decision
Register `GLAConfig` and `GLAModel` directly into `transformers` auto-registries.

### Rationale
Enables seamless compatibility with HuggingFace Hub, Pipelines, and Trainer.

### Consequences
**Easier:**
- Standard `AutoModelForCausalLM.from_pretrained` works out-of-the-box.
**Harder:**
- Requires matching exact HF attribute naming conventions.

### Alternatives Considered
- **Alt 1: Standalone custom wrapper class** — Rejected because HF pipelines cannot instantiate it.
- **Alt 2: Subclassing GPT2LMHeadModel** — Rejected because attention signature differences break GPT-2 block assumptions.
- **Alt 3: HuggingFace fork** — Rejected due to maintenance overhead.

---

## ADR-008: Default Chunk Size of 64
### Context
Chunked GLA parallelizes state accumulation across chunks of length $C$.

### Decision
Set default chunk size $C=64$.

### Rationale
Balances tensor core SRAM utilization, register pressure, and sequence parallelism.

### Consequences
**Easier:**
- High throughput on modern NVIDIA GPUs (RTX 30xx/40xx, A100, H100).
**Harder:**
- Sequences shorter than 64 require padding or dynamic block handling.

### Alternatives Considered
- **Alt 1: Chunk size 32** — Rejected due to underutilization of Tensor Core matrix multiply units.
- **Alt 2: Chunk size 128** — Rejected because intra-chunk $O(C^2)$ computation increases SRAM register spills.
- **Alt 3: Pure token-by-token recurrence** — Rejected because lack of parallelism severely degrades GPU utilization.

---

## ADR-009: Log-Space Gating Computation
### Context
Cumulative product of sigmoid gates $\prod \alpha$ causes underflow or overflow in long contexts.

### Decision
Compute gates in log-space via `F.logsigmoid(logits / tau)`.

### Rationale
Ensures numeric stability across arbitrary sequence lengths without intermediate zero-underflow.

### Consequences
**Easier:**
- Zero NaNs or Infs across sequence lengths up to 8k+ tokens.
**Harder:**
- Requires working with log-space subtractions and cumulative sums.

### Alternatives Considered
- **Alt 1: Raw sigmoid followed by product** — Rejected because it underflows to zero on long sequences.
- **Alt 2: Clamped sigmoid** — Rejected because gradient vanishes outside clamped bounds.
- **Alt 3: Softplus gating** — Rejected because decay factor is not bounded in $[0, 1]$.

---

## ADR-010: Single Scale Factor Application
### Context
GLA inner products require attention scale $1/\sqrt{d}$ to prevent dot product explosion.

### Decision
Apply scale strictly once to query $Q$ (`q = q * self.scale`).

### Rationale
Prevents duplicate scaling between intra-chunk and inter-chunk reduction paths.

### Consequences
**Easier:**
- Consistent scaling across all chunk boundaries and state updates.
**Harder:**
- Must ensure Triton and fallback kernels do not re-apply the scale internally.

### Alternatives Considered
- **Alt 1: Split scale $\frac{1}{d^{1/4}}$ on Q and K** — Rejected because state update $k^T v$ becomes scale-dependent.
- **Alt 2: Scale applied to final output** — Rejected because attention score decay weights are distorted.
- **Alt 3: Scale applied inside both reduction loops** — Rejected because it resulted in historical $8\times$ magnitude error.
