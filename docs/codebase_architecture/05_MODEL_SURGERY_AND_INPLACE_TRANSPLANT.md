# NeuralOps Codebase Architecture: Part 05
# Model Surgery, In-Place Transplant & Modular Surgical Deltas

**Assigned Subagent**: `Agent 05: Model Surgery & Subspace Grafting Specialist`
**Target Codebase Paths**:
- `neuralops/surgery/`
- `neuralops/surgery/in_place.py`
- `neuralops/models/`
- `neuralops/models/surgical_io.py`
- `neuralops/surgery/reconciliation.py`

---

## 1. Executive Summary & Design Philosophy

Upgrading pretrained large language models from quadratic attention ($\mathcal{O}(N^2)$) to linear, state-space, or hybrid attention traditionally required retraining foundation models from scratch at immense computational expense.

Part 05 details the **Model Surgery Engine** in `neuralops/surgery/` and the **Modular Surgical Delta Protocol** (`neuralops.models.surgical_io`). This subsystem performs zero-cost, in-place attention transplantation on arbitrary pretrained PyTorch/HuggingFace transformer architectures:
- **Recursive AST Traversal**: Automatically detects and isolates transformer attention blocks (`q_proj`, `k_proj`, `v_proj`, `o_proj`, or combined `qkv_proj`).
- **In-Place Module Grafting**: Replaces target attention layers with efficient equivalents from `ATTENTION_REGISTRY` via `replace_attention()`.
- **Modular Surgical Delta Protocol**: Rather than persisting redundant 440 MB to 16 GB base models across hundreds of attention variants, serializes only modified parameters into `attention_weights.safetensors` alongside a structured `surgical_manifest.json` (**97%–99.4% footprint reduction**).
- **Weight Reconciliation**: Splits or tiles projection weights, aligns head dimensions, and ensures exact preservation of positional embeddings (RoPE, ALiBi) and causal attention masking contracts.

---

## 2. Assigned Subagent Responsibilities (Agent 05)

Agent 05 is responsible for:
1. **Zero-Copy Parameter Grafting**: Guaranteeing that parameter grafting maintains existing memory pointers where possible or allocates aligned contiguous blocks without memory leaks.
2. **Projection Alignment**: Handling fused-QKV projections (`qkv_proj` $\in \mathbb{R}^{3D \times D}$) and Grouped-Query Attention (GQA) head expansions seamlessly.
3. **Shape Invariant Preservation**: Enforcing the strict tensor contract:
   $$\text{Input Shape } (B, T, D) \equiv \text{Output Shape } (B, T, D)$$
4. **Modular Surgical Delta Export**: Guaranteeing that every surgical run exports valid `attention_weights.safetensors` and `surgical_manifest.json`, reconstituted in sub-450 ms via `load_surgeried_model()`.
5. **Manifest Generation & AGENTS.md §7 Compliance**: Ensuring operations mutate models solely through defined protocols and transactions, never violating layer boundaries.

---

## 3. Core File Inventory & Symbol Mapping

| File Path | Core Symbols / Functions | Architectural Responsibility |
|---|---|---|
| `neuralops/surgery/in_place.py` | `replace_attention`, `transplant_module`, `revert_surgery` | In-place module substitution, hook manipulation, and AST replacement |
| `neuralops/surgery/reconciliation.py` | `reconcile_qkv_weights`, `expand_gqa_heads`, `align_rope_dim` | Weight tensor slicing, shape transformation, and rotary dimension adaptation |
| `neuralops/surgery/manifest.py` | `SurgeryManifest`, `SurgeryRecord`, `compute_model_diff` | Cryptographic parameter hashing, state serialization, and reproducibility logging |
| `neuralops/models/transformer_wrapper.py` | `NeuralOpsModel`, `wrap_pretrained_model` | Uniform forward-pass execution wrapper isolating base backbone details |

---

## 4. Algorithmic Steps of In-Place Transplant

```
[Pretrained Model (e.g. LLaMA / GPT-2)]
                 │
                 ▼
      [Step 1: AST Discovery]
   (Inspect named_modules recursively)
                 │
                 ▼
    [Step 2: Weight Reconciliation]
   (Extract Q/K/V weights, resolve GQA/RoPE)
                 │
                 ▼
  [Step 3: Graft New Attention Module]
   (Instantiate operator from ATTENTION_REGISTRY)
                 │
                 ▼
   [Step 4: Subspace Ridge Alignment]
   (Solve Fisher-Ridge projection for W_O)
                 │
                 ▼
     [Step 5: Manifest Generation]
   (Record SHA-256 hashes, FLOPs, and parameter diffs)
```

---

## 5. Detailed Implementation Blueprint

### 5.1 Fused QKV Slicing Protocol
When grafting an architecture where Q, K, and V are stored in a single combined projection $\mathbf{W}_{\text{qkv}} \in \mathbb{R}^{(d_q + d_k + d_v) \times D}$:
```python
def reconcile_qkv_weights(orig_layer, target_attn):
    if hasattr(orig_layer, "qkv_proj"):
        qkv = orig_layer.qkv_proj.weight.data
        dq, dk, dv = target_attn.q_dim, target_attn.k_dim, target_attn.v_dim
        target_attn.q_proj.weight.data.copy_(qkv[:dq, :])
        target_attn.k_proj.weight.data.copy_(qkv[dq:dq+dk, :])
        target_attn.v_proj.weight.data.copy_(qkv[dq+dk:, :])
    elif hasattr(orig_layer, "q_proj"):
        target_attn.q_proj.weight.data.copy_(orig_layer.q_proj.weight.data)
        target_attn.k_proj.weight.data.copy_(orig_layer.k_proj.weight.data)
        target_attn.v_proj.weight.data.copy_(orig_layer.v_proj.weight.data)
```

### 5.2 Rotary Embedding Compatibility Check
The transplant engine explicitly validates rotary embedding head dimension $d_h$:
$$\text{head\_dim}_{\text{target}} \equiv \text{head\_dim}_{\text{source}}$$
If dimensions diverge, an explicit projection layer is inserted with an automated warning emitted.

### 5.3 Modular Surgical Delta Serialization & Reconstitution
Post-transplant parameters are exported using `neuralops.models.surgical_io`:
```python
from neuralops.models.surgical_io import save_surgeried_model, load_surgeried_model

# 1. Export delta tensors and topological manifest (cuts 440MB to ~10MB)
save_surgeried_model(
    model=surgeried_model,
    output_dir="results/bert_attention_zoo/flash_attention",
    base_model_name="bert-base-uncased",
    attention_type="flash_attention",
)

# 2. Dynamic runtime reconstitution (< 450 ms)
reconstituted_model = load_surgeried_model("results/bert_attention_zoo/flash_attention")
```

---

## 6. Audit & Safety Verification Contract

- **Test Suite**: `tests/surgery/test_in_place_transplant.py`, `tests/architecture/test_import_boundaries.py`
- Verification standard:
  1. Forward pass tensor shape identity verified for sequence lengths $T \in \{128, 512, 2048, 8192\}$.
  2. Loss evaluation immediately post-surgery (Step 0) must remain bounded ($\text{Loss} \le 3 \times \text{Baseline}$).
  3. Reversion test: calling `revert_surgery(model, manifest)` must restore original weights with 0 numerical divergence.
  4. Modular delta verification: `attention_weights.safetensors` size must be $< 15\%$ of base model size ($> 85\%$ empirical compression, targeting $>97\%$).
  5. AGENTS.md §7 compliance: surgery modules must never directly invoke CLI, runners, or persistence backends.
