# 🧠 Efficient Attention Modules for DistilBERT & Hybrid Models

A curated collection of **30+ state‑of‑the‑art attention variants** – from zero‑shot compression suites to hardware‑optimised kernels – all ready to drop into DistilBERT (and similar models) via a unified registry.

Every module follows the standard `(hidden_states, attention_mask) -> (output, attn_weights)` interface.

## 📁 Folder Structure

```
attentions/
├── __init__.py                    # Auto‑imports all sub‑modules
├── core/
│   ├── base.py                    # BaseAttention class
│   └── registry.py                # ATTENTION_REGISTRY
│
├── aurora.py                      # Full AURORA (10‑in‑1)
├── minimal_aurora.py              # MinimalAURORA (SVD only)
├── nova.py                        # NOVA full
├── minimal_nova.py                # NOVA minimal
├── orion.py                       # ORION hybrid
├── phantom.py                     # PHANTOM (CUR + LSH)
├── prism_full.py                  # PRISM full (6 techniques)
├── minimal_prism.py               # PRISM minimal
├── optimized_hybrid_attention.py  # OHA (sliding window + low‑rank)
│
├── adasplash_attention.py         # AdaSplash (entmax sparse)
├── adaptive_span_attention.py     # Per‑head learnable span
├── block_transformer_hiba.py      # Block Transformer HIBA
├── cluster_attention.py           # k‑means clustering
├── flash_attention3.py            # FlashAttention3
├── flash_attention3_fp8.py        # FlashAttention3 FP8 block
├── full_attention.py              # Standard BERT attention
├── hamming_attention.py           # Hamming distance attention
├── hla_attention.py               # Hadamard Linear Attention
├── infini_attention.py            # Infini‑Attention
├── linformer_attention.py         # Linformer
├── lsh_attention.py               # Reformer LSH
├── mixattention_moska.py          # MixAttention MoSKA
├── moba_attention.py              # Mixture of Blocks (MoBA)
├── nha_attention.py               # Native Hybrid Attention
├── nsa_attention.py               # Native Sparse Attention
├── routing_attention.py           # Routing Transformer
├── span_expanded_attention.py     # Span expanded + LoRA
├── vig_attention.py               # ViG (vision GNN‑style)
├── xattention.py                  # XAttention (antidiagonal block sparse)
│
├── (grouped_query/...)            # GQA, MLA, QCQA, WGQA
├── (ssm_hybrid_blocks...)         # Mamba‑based hybrid blocks
└── README.md
```

---

## 🔌 Registry Keys

Most modules are registered in `ATTENTION_REGISTRY` with a unique key.  
You can inject them via:

```python
from framework.attentions.core.registry import ATTENTION_REGISTRY
ATTENTION_REGISTRY["key"](layer, model_config, comp_config, ...)
```

| Module | Registry Key |
|--------|--------------|
| AURORA | `aurora` |
| MinimalAURORA | `minimal_aurora` |
| NOVA | `nova` |
| MinimalNOVA | `minimal_nova` |
| PRISM Full | `prism_full` |
| MinimalPRISM | `minimal_prism` |
| ORION | `orion` |
| PHANTOM | `phantom` |
| OHA | `oha` |
| Adaptive Span | `adaptive_span` |
| BlockTransformer HIBA | `block_transformer_hiba` |
| Cluster | `cluster` |
| LSH | `lsh` |
| MixAttention MoSKA | `mixattention_moska` |
| MoBA | `moba` |
| NSA | `nsa` |
| Routing | `routing` |
| XAttention | `xattention` |
| FlashAttention3 | `flash3` |
| FullAttention | `full` |
| Hamming | `hamming` |
| HLA | `hla` |
| Infini | `infini` |

(Some modules like `AdaSplash`, `ViG`, `Linformer` are not registered by default; they can be used standalone.)

---

## 🧩 Compressed Full‑Attention Suites

These modules replace the standard Q/K/V/Out projections and optionally the attention mechanism to dramatically reduce parameters and memory.

| Module | Techniques | Zero‑Shot | Parameter Reduction |
|--------|------------|-----------|---------------------|
| **MinimalAURORA** | SVD factorisation only | ✅ | 2‑4× |
| **AURORA** | SLiM, RPCA, OATS, SliceGPT, NMF, ProxSparse, DeltaKV, KVTC, Softpick, Dual pathway | Partially | up to 10× |
| **MinimalNOVA** | SVD factorisation | ✅ | 2‑4× |
| **NOVA** | ARMOR, AST, DeltaKV, KVT, Softpick, Subspace | ✅ | 10‑20× |
| **MinimalPRISM** | SVD factorisation | ✅ | 2‑4× |
| **PRISM Full** | ARMOR, AST, DeltaKV, KVT, Softpick, SVD rank selection | ✅ | 10‑20× |
| **ORION** | SVD + head‑split (kernel/full) + Softpick | ✅ | 2‑4× |
| **PHANTOM** | CUR low‑rank + LSH attention + Softpick | ✅ | 5‑20× |
| **OHA** | SVD + sliding window local + low‑rank global + gate fusion | ✅ | 2‑4× |

**Usage (injection):**
```python
from framework.attentions.minimal_aurora import inject_minimal_aurora
inject_minimal_aurora(layer, model_config, comp_config, layer_idx, total_layers, old_attn)
```

Many suites provide their own dataclass config (e.g., `AURORAConfig`, `NOVAConfig`).

---

## 🎯 Sparse & Adaptive Attention

These modules do **not** change projection parameters, but they modify the attention pattern to reduce compute and memory.

| Module | Method | Complexity | Trainable Sparsity |
|--------|--------|------------|-------------------|
| **AdaptiveSpan** | Each head learns a soft attention span | O(N²) with soft truncation | Yes (span per head) |
| **BlockTransformer HIBA** | Intra‑block full + inter‑block sparse + periodic rectification | ~O(N·B + N²/interval) | Partially |
| **ClusterAttention** | k‑means token clustering | O(N·K + (N/K)²) | Implicit via clustering |
| **LSHAttention** | Multi‑round angular LSH | O(N·B·rounds) | Hash‑based |
| **MixAttentionMoSKA** | FASA token pruning | O(N·keep) | Yes (keep ratio) |
| **MoBAAttention** | Block‑wise top‑k selection | O(N·blocks) | Yes |
| **NSA** | Local window + global top‑k | O(N·W + N·K) | Yes |
| **RoutingAttention** | Online k‑means routing | O(N·K + (N/K)²) | Implicit |
| **SpanExpanded** | Fixed expansion KV cache + LoRA | O(N·C) with cache | No (cache size fixed) |
| **XAttention** | Antidiagonal block sparse | O(N·B·keep) | Yes (keep fraction) |

**Usage:**
```python
# Direct instantiation
from framework.attentions.cluster_attention import ClusterAttention
attn = ClusterAttention(config, num_clusters=64)
out, _ = attn(hidden_states, attention_mask)
```

**Note:** Some sparse modules (especially the clustering‑based ones) work better during inference; training may require extra warm‑up or distillation.

---

## ⚡ Fast & Hybrid Modules

Ultra‑fast attention blocks using hardware‑friendly operations, linear attention, or binary codes.

| Module | Technique | Complexity | Key Benefit |
|--------|-----------|------------|-------------|
| **AdaSplash** | α‑entmax sparse attention | O(N²) | Sparse without fixed patterns |
| **FlashAttention3** | Tiled GPU kernel (FA3), optional FP8 | O(N²) (tiled) | 2‑3× faster, low memory |
| **FlashAttention3 FP8** | FA3 with FP8 storage | O(N²) | Further memory savings |
| **FullAttention** | Standard BERT attention | O(N²) | Baseline |
| **HammingAttention** | Binary codes + Hamming distance | O(N²) with binary ops | Multiply‑free, fast |
| **HLAAttention** | Hadamard linear attention | O(N) | Linear in seq length |
| **InfiniAttention** | Compressive memory + local window | O(N·W + N·C) | Unlimited context |
| **LinformerAttention** | Low‑rank K/V projection | O(N·k) | Linear complexity |
| **NHA (Native Hybrid)** | Mix of full & linear attention | O(N²·p + N) | Speed‑accuracy trade‑off |
| **ViGAttention** | Direction‑wise gated linear attention | O(N·D) | Efficient for 2D data |

**Usage (standalone):**
```python
from framework.attentions.flash_attention3 import FlashAttention3
attn = FlashAttention3(config, use_fp8=True)
out, _ = attn(hidden_states)
```

**Notes:**
- For **FlashAttention3** and **FP8** blocks, you need `flash_attn>=2.5` and optionally `flash_attn_interface`.
- **AdaSplash** and **Hamming** produce sparse attention maps directly.
- **HLAAttention** and **Linformer** completely avoid the quadratic bottleneck.

---

## ⚙️ Configuration

Many modules accept a standard HuggingFace‑style `config` object (containing `hidden_size` and `num_attention_heads`).  
The compression suites also accept a `comp_config` that maps to their internal `*Config` dataclass.

### Example: Configuring Full PRISM

```python
from framework.attentions.prism_full import PRISMConfig, PRISMAttention

cfg = PRISMConfig(
    energy_threshold=0.95,
    ast_enabled=True,
    ast_sparsity=0.5,
    deltakv_enabled=True,
    kvt_enabled=True,
    softpick_enabled=True,
)
new_attn = PRISMAttention(model_config, cfg, pretrained_attn=old_attn)
```

### Example: Using FlashAttention3 with FP8

```python
from framework.attentions.flash_attention3_fp8 import FlashAttention3FP8Block

block = FlashAttention3FP8Block(d_model=768, n_heads=12, causal=False, use_fp8=True)
out = block(x)
```

---

## 📊 Comparison Dashboard

| Module | Param Reduction | Attention Complexity | Memory (KV cache) | Zero‑Shot | Trainable Sparsity |
|--------|----------------|----------------------|------------------|-----------|-------------------|
| FullAttention | – | O(N²) | O(N²) | ✅ | ❌ |
| MinimalAURORA | 2‑4× | O(N²) | O(N²) | ✅ | ❌ |
| Full AURORA | 4‑10× | O(N²) (with sparse) | O(N) (DeltaKV) | Partially | ✅ |
| NOVA | 10‑20× | O(N²) (with sparse) | O(N) (KVT) | ✅ | ✅ |
| PRISM Full | 10‑20× | O(N²) (with sparse) | O(N) | ✅ | ✅ |
| PHANTOM | 5‑20× | O(N·B·K) (LSH) | O(N·B) | ✅ | LSH‑based |
| ORION | 2‑4× | Hybrid | O(N·F + N²) | ✅ | Softpick |
| OHA | 2‑4× | O(N·W + N·R) | O(N·R) | ✅ | ❌ |
| AdaptiveSpan | – | O(N²) (soft‑capped) | O(N²) | ❌ | ✅ |
| LSH | – | O(N·B·R) | O(N·B) | ❌ | Hash‑based |
| BlockTransformer | – | ~O(N·B) | O(N·B) | ✅ (fixed) | ❌ |
| Routing | – | O((N/K)²) | O((N/K)²) | ❌ | Implicit |
| NSA | – | O(N·W + N·K) | O(N·K) | ✅ | ✅ |
| XAttention | – | O(N·B·keep) | O(N·B·keep) | ✅ | ✅ |
| AdaSplash | – | O(N²) | O(N²) | ✅ | α‑entmax |
| Flash3 | – | O(N²) tiled | O(N²) | ✅ | ❌ |
| Hamming | – | O(N²) binary | O(N²) | ✅ | ✅ |
| HLA | – | O(N) | O(N) | ✅ | ❌ |
| Infini | – | O(N·W + N·C) | O(N·C) | ✅ | ❌ |
| Linformer | – | O(N·k) | O(N·k) | ✅ | ❌ |
| NHA | – | O(N²·p + N) | O(N²·p) | ✅ | ❌ |
| ViG | – | O(N·D) | O(N) | ✅ | ❌ |

*All metrics assume encoder‑style attention (no causal masking for some).*

---

## 🧪 Adding a New Module

1. Create a `.py` file in this folder.
2. Subclass `nn.Module` (or `BaseAttention`).
3. Implement `forward(self, hidden_states, attention_mask=None, ...)` returning `(output, attention_weights)`.
4. Decorate with `@ATTENTION_REGISTRY.register("your_key")`.
5. Add `from . import your_module` to `__init__.py` (or auto‑generate it).

---

## 📜 References & Credits

This collection implements or adapts ideas from:

- AURORA, NOVA, PRISM, ORION, PHANTOM, OHA – original efficient attention suites.
- FlashAttention‑3, Reformer (LSH), Routing Transformer, Adaptive Span, Linformer.
- Infini‑Attention, NSA, MoBA, XAttention, ViG, Hamming (HAD), AdaSplash.
- Native Hybrid Attention (NHA), HLA, GQA, MLA, QCQA, WGQA.
- Various hybrid SSM blocks (GoldFinch, GRiffin, Hymba, Jamba, Samba …)

All implementations are provided for research and production use as part of the **Efficient Transformer Toolkit**.

---

*Happy compressing! 🚀*
