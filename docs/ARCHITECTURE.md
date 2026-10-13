# NeuralOps Architectural Specification (v1.3.0)

NeuralOps is architected as a modular, industrial ML systems library designed to enable drop-in attention surgery, deep model compression, high-performance Triton kernel execution, and resilient multi-task benchmarking.

---

## 1. System Hierarchy & Component Topography

The platform is organized into 10 canonical public subpackages, anchored by the high-performance `neuralops.core` engine:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      Unified Public API (neuralops.api)                   │
│   replace_attention(), compress(), export_model(), load_model(), info()   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│  attention   │              │ compression  │              │   kernels    │
│  (131 Mechs) │              │  (72 Engines)│              │  (26 Modules)│
└──────┬───────┘              └──────┬───────┘              └──────┬───────┘
       │                             │                             │
       └─────────────────────────────┼─────────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │        neuralops.core        │
                      │  - _attention_surgery.py     │
                      │  - compression_pipeline.py   │
                      │  - triton_kernels/op_reg.py  │
                      │  - architecture.py           │
                      └──────────────┬───────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│    models    │              │   training   │              │  benchmarks  │
│ HF AutoModel │              │  Distill/AMP │              │ Orchestrator │
└──────────────┘              └──────────────┘              └──────────────┘
```

---

## 2. Attention Mechanism Taxonomy (8 Canonical Families)

NeuralOps unifies 131 distinct attention mechanisms across 8 mathematical families:

1. **Linear / Kernel Attention**: $O(N)$ complexity through feature map kernel formulations ($\phi(Q)\phi(K)^T V$). Includes Cosformer, GLA, Linformer, Performer, Galerkin, RALA, Primal.
2. **Gated / DeltaNet**: Recurrent state-tracking models with input-dependent gates and linear recurrence. Includes GDN, HGRN2, KDA, HLA, DecayLinear.
3. **SSM / Recurrent**: State Space Models and selective recurrent scans. Includes Mamba-1, Mamba-2, Selective SSM, FMS, Griffin, Jamba blocks.
4. **Sparse / Routing**: Block-sparse, cluster, and hash-based token selection. Includes MOBA, NSA, Routing Attention, Cluster Attention, LSH, Hamming Attention.
5. **Hybrid / Sliding Window**: Blended multi-branch mechanisms combining local window attention with global linear or recurrent states. Includes Aegis, Regla, Landmark, Split-Head, MixAttention.
6. **SVD / Factorized**: Low-rank subspace projections on query/key/value states. Includes SVD-Linear, ASVD, SVD-LLM, Residual-SVD, LoRA-Attention.
7. **Modern / GQA**: Production architectures featuring multi-query, grouped-query, and multi-head latent attention. Includes GQA, MLA, WGQA, QCQA, DynamicConv.
8. **Adaptive / Dynamic**: Dynamic token compression and adaptive span mechanisms. Includes Infini-Attention, Lola, Sink Attention, Token-Compressed Attention.

---

## 3. Tensor Flow and Attention Surgery Pipeline

When `neuralops.replace_attention(model, "module_name")` is invoked, the surgical injection pipeline performs the following transformations:

```mermaid
sequenceDiagram
    participant User as Caller / Application
    participant Surgery as Attention Surgery Engine
    participant Model as Target Transformer Layer
    participant Wrapper as Surgery Layer Wrapper
    participant Kernel as Triton / Fallback Kernel

    User->>Surgery: replace_attention(model, "cosformer")
    Surgery->>Model: Inspect module hierarchy (q_proj, k_proj, v_proj, o_proj)
    Surgery->>Model: Extract weights and detect RoPE / GQA configuration
    Surgery->>Wrapper: Instantiate custom attention module with transplanted weights
    Surgery->>Model: Swap self_attn attribute in-place
    Surgery->>Model: Append metadata entry to model._neuralops_metadata

    User->>Model: Forward pass: hidden_states (B, S, D)
    Model->>Wrapper: forward(hidden_states, attention_mask, position_ids, ...)
    Wrapper->>Kernel: triton_dispatch("linear_attn", Q, K, V)
    Kernel-->>Wrapper: Attn Output (B, S, D)
    Wrapper-->>Model: Output Tuple (attn_output, past_key_values)
```

---

## 4. Triton Kernel Architecture & `torch.compile` Compatibility

NeuralOps provides 26 fused Triton kernel modules in `neuralops.core.triton_kernels`. To maintain graph continuity during `torch.compile(model)` execution without introducing graph breaks:

1. **Custom Op Registration**: Operations are registered in the PyTorch dispatcher via `torch.library.custom_op` under the `neuralops::` namespace (e.g., `neuralops::linear_attn`, `neuralops::gdn_scan`).
2. **FakeTensor Shape Propagation**: Each custom op registers a `.register_fake` handler enabling symbolic shape deduction without GPU tensor execution during dynamo tracing.
3. **Transparent Fallback**: When running on CPU or unsupported precisions (e.g. float32 on GPUs without tensor cores), `triton_dispatch` routes calculations to differentiable, numerically aligned PyTorch fallbacks.

---

## 5. Model Compression & Production Export Pipeline

The compression engine operates through structured factorization passes:
- **Low-Rank Factorization**: Decomposes weight matrix $W \in \mathbb{R}^{m \times n}$ into $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{r \times n}$.
- **Adaptive SVD (ASVD)**: Allocates per-layer ranks dynamically based on singular value energy distribution.
- **Export & Round-Trip**: Saves model weights to `model.safetensors` alongside `neuralops_config.json`. The `load_model()` factory reconstructs the exact layer substitutions and parameter mappings on reload.
