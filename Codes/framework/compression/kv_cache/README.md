# 🗄️ KV‑Cache & Advanced Weight Compression

This sub‑package implements **five state‑of‑the‑art compressors** targeting the KV‑cache and weight matrices of Transformer models, as described in ICLR 2026, EACL 2026, and arXiv 2026.  
They reduce memory usage, speed up inference, and can be combined with other compression methods.

## 📦 Available Compressors

| Registry Key | Class | Technique | Context |
|--------------|-------|-----------|---------|
| `zerokv_neo` | `ZeroKVNeoCompressor` | L2‑norm eviction + attention sinks | KV‑cache / weight pruning |
| `kvtc` | `KVTCCompressor` | PCA decorrelation + quantisation + entropy coding | KV cache / weight compression |
| `hybridkv` | `HybridKVCompressor` | Static/dynamic head classification | Multimodal LLM KV cache |
| `delta_kv` | `DeltaKVCompressor` | Residual‑based compression | KV cache / weight low‑rank + sparse |
| `cachenotes` | `CacheNotesCompressor` | Task‑aware compression via CPTs | Task‑specific KV cache |

---

## 🚀 Quick Start

### Compress via Registry

```python
from framework.compression import COMPRESSOR_REGISTRY

# Heavy‑hitter eviction (keep 30% of attention neurons)
compressor = COMPRESSOR_REGISTRY.get("zerokv_neo")(model, keep_ratio=0.3)
model, stats = compressor.compress()

# PCA + quantisation (4‑bit, 95% energy)
compressor = COMPRESSOR_REGISTRY.get("kvtc")(model, bits=4, pca_energy=0.95)
model, stats = compressor.compress()
```

### Compress via CLI

```bash
python cli.py compress --method zerokv_neo --keep-ratio 0.3
python cli.py compress --method kvtc --bits 4
python cli.py compress --method hybridkv --budget 0.3
python cli.py compress --method delta_kv --rank 16 --sparsity 0.29
python cli.py compress --method cachenotes --task-type qa --keep-ratio 0.3
```

---

## 🧬 Detailed Descriptions

### 1. ZeroKV‑Neo – Heavy‑Hitter Eviction (`zerokv_neo`)
- **Inspiration:** *”A Simple and Effective L2 Norm‑Based Strategy for KV Cache Compression”* + H2O (Heavy‑Hitter Oracle).
- **How it works:**  
  For each attention projection (Q/K/V), computes the L2 norm of output neurons, then retains a fixed number of **attention sinks** (top norms) plus additional **heavy hitters** (top‑k by norm).  
  Effectively prunes less important neurons, mimicking KV‑cache eviction with attention sink preservation.
- **Parameters:**
  - `keep_ratio` – fraction of total neurons to retain (default 0.3).
  - `num_attention_sinks` – initial high‑norm neurons always kept (default 4).
  - `heavy_hitter_ratio` – extra fraction of heavy hitters on top of sinks (default 0.1).

### 2. KVTC – Transform Coding (`kvtc`)
- **Paper:** *”KV Cache Transform Coding for Compact Storage in LLM Inference”* – Staniszewski & Lancucki, ICLR 2026.
- **How it works:**  
  Applies **PCA decorrelation** (via SVD on the weight matrix), retains a given fraction of the spectral energy, then **quantises** the low‑rank approximation to `bits` bits.  
  Additionally estimates a Huffman‑like entropy coding compression ratio.
- **Parameters:**
  - `bits` – quantisation bit‑width (default 4).
  - `pca_energy` – fraction of singular‑value energy retained (default 0.95).
- **Output stats include:** `avg_entropy_compression_ratio` in addition to standard compression metrics.

### 3. HybridKV – Static/Dynamic Head Classification (`hybridkv`)
- **Paper:** *”HybridKV: Hybrid KV Cache Compression for Efficient MLLM Inference”* – Zeng et al., arXiv:2604.05887.
- **How it works:**  
  Classifies each attention head as **static** (low variance across output) or **dynamic** (high variance).  
  Static heads are pruned uniformly by keeping top rows by L2 norm; dynamic heads use a **chunk‑wise top‑k** retention to preserve variety.  
  A hierarchical budget splits the overall retention ratio between static and dynamic heads.
- **Parameters:**
  - `budget` – overall fraction of parameters to retain (default 0.3).
  - `static_ratio` – fraction of the budget given to static heads (default 0.7).

### 4. DeltaKV – Residual Compression (`delta_kv`)
- **Paper:** *”DeltaKV: Residual‑Based KV Cache Compression via Long‑Range Similarity”* – Hao et al., arXiv:2602.08005.
- **How it works:**  
  Decomposes each weight matrix into a **low‑rank reference** (SVD) plus a **sparse residual**.  
  Only the top `sparsity` fraction of residual entries are kept, emulating the idea of storing semantic residuals relative to historical references (the original DeltaKV paper operates on K/V tensors directly; here we adapt the principle to weight matrices).
- **Parameters:**
  - `rank` – rank of the reference low‑rank component (default 16).
  - `sparsity` – fraction of residual entries to retain (default 0.29, matching the original 29% storage).
- **Also provides:** `DeltaKVStore` – a standalone class for real‑time KV cache compression during inference.

### 5. CacheNotes – Task‑Aware Compression (`cachenotes`)
- **Paper:** *”CacheNotes: Task‑aware key‑value cache compression for reasoning‑intensive knowledge tasks”* – Corallo et al., EACL 2026.
- **How it works:**  
  Given a task type (e.g., `qa`, `summarize`, `code`, `reasoning`), generates **Compression‑Planning Tokens (CPTs)** – importance scores per output neuron.  
  These scores are combined with weight magnitudes to decide which neurons to keep.
- **Parameters:**
  - `task_type` – type of downstream task.
  - `keep_ratio` – fraction of parameters to retain (default 0.3).

---

## ⚙️ Common Parameters

| Parameter | Typical default | Meaning |
|-----------|-----------------|---------|
| `keep_ratio` | 0.3 | Fraction of neurons/entries retained |
| `bits` | 4 or 8 | Quantisation bit‑width |
| `rank` | 16–64 | Low‑rank approximation rank |
| `sparsity` | 0.29 | Fraction of sparse residual kept |
| `budget` | 0.3 | Total retention ratio (HybridKV) |
| `task_type` | `"qa"` | Downstream task identifier |

---

## 📊 Comparison

| Compressor | Best for | Memory reduction | Speed impact | Requires training? |
|------------|----------|------------------|--------------|-------------------|
| ZeroKV‑Neo | Quick heavy‑hitter pruning | ~70% | No overhead | ❌ (post‑hoc) |
| KVTC | Extreme storage reduction | 8‑16× with quant | Minimal | ❌ |
| HybridKV | Multimodal LLMs | ~70% | Minimal | ❌ |
| DeltaKV | Preserving fidelity | ~70% + low‑rank | Slight | ❌ |
| CacheNotes | Task‑specific deployment | ~70% | None | ❌ |

All compressors are post‑training (no fine‑tuning required), except they may need a few calibration samples (KVTC uses the weights only).

---

## 🔌 Integration

- All inherit from `BaseCompressor`, return `(model, stats)`.
- `COMPRESSOR_REGISTRY` provides keys for CLI and scripting.
- `DeltaKVStore` can be used independently in inference pipelines for real‑time K/V compression.

---

## 📜 References

- **ZeroKV‑Neo**: H2O / L2‑norm KV cache eviction.
- **KVTC**: Staniszewski & Lancucki, ICLR 2026.
- **HybridKV**: Zeng et al., arXiv:2604.05887.
- **DeltaKV**: Hao et al., arXiv:2602.08005.
- **CacheNotes**: Corallo et al., EACL 2026.

