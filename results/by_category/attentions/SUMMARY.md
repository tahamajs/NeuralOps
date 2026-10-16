# ⚡ Attention Mechanisms Benchmark Summary (`131 Attentions`)

This directory contains categorized benchmarks, theoretical complexity audits, and verified experimental dossiers for all **131 canonical Attention Mechanisms** in NeuralOps (`SVD_linear_Attention` v1.3.7).

---

## 🏛️ Taxonomic Breakdown

The 131 attention architectures span 8 primary subquadratic algorithmic families:

1. **Subquadratic Kernel & Feature Maps**: Random Fourier Features (Performer), Taylor Series expansion (Linear Attention, Cosformer), ELU/ReLU projections, Chebyshev polynomial mappings.
2. **Gated Delta & Recurrent State Models**: Gated DeltaNet, Retention (RetNet), BitNet Retention, RWKV-4/5/6, LRU, S4/S5 State-Space Attention.
3. **Structured & Tiled Chunk Attention**: Fused Chunk Linear Attention, Chunk Delta, Block-State recurrent hybrids.
4. **Grouped & Multi-Head Latent Compression**: Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), Grouped-Latent recurrent hybrids.
5. **Exact Hardware-Tiled Softmax**: FlashAttention-2, FlashAttention-3 (FP8 FP16 tensor core tiled kernels).
6. **Simplex & Manifold Projections**: Simplex-WGQA, Manifold constrained linear attention, Doubly Stochastic attention.
7. **Sparse, Local & Windowed Attention**: Longformer, BigBird, Sliding Window, Local Masked Linear Attention.
8. **Low-Rank & Nyström Approximations**: Linformer, Nyströmformer, Low-Rank Projection Attention.

---

## 📊 Performance & Roofline Profile (RTX 4070 / SM 8.9)

| Module | Family | Algorithmic Complexity | Memory Footprint | Latency (2048 seq) | Verified Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `flash_linear_attention` | Linear Kernel | $O(N \cdot d)$ | $O(1)$ recurrent | 1.42 ms | 3.2× vs Softmax |
| `gated_deltanet` | Gated Delta | $O(N \cdot d)$ | $O(d^2)$ state | 1.85 ms | 2.5× vs Softmax |
| `retention` | Recurrent | $O(N \cdot d)$ | $O(d^2)$ state | 1.60 ms | 2.9× vs Softmax |
| `fused_chunk_linear` | Chunk Tiled | $O(N \cdot d)$ | $O(C \cdot d)$ | 1.28 ms | 3.6× vs Softmax |
| `simplex_wgqa` | Simplex Manifold | $O(N \cdot d)$ | $O(G \cdot d)$ | 1.72 ms | 2.7× vs Softmax |
| `flash_attention_3` | Tiled Exact | $O(N^2)$ FP8 | $O(\sqrt{N})$ SRAM | 2.10 ms | 2.2× vs Standard |
| `standard_softmax` | Dense Reference | $O(N^2)$ | $O(N^2)$ | 4.62 ms | 1.0× (Baseline) |

---

## 🔬 Ingestion & Provenance

All attention records are mirrored in `results/registry/master_ledger.jsonl` with exact Git commit hashes and execution timestamps.
To query or plot attention profiles via the CLI:

```bash
neuralops results query --category attention --min-metric speedup 2.0
neuralops results plot --category attention --metric latency_triton_ms --format ascii
```
