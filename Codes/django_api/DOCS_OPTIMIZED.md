# 📚 Efficient Transformer Toolkit – Technical Documentation

This document provides a comprehensive overview of the optimized attention mechanisms and compression techniques available in the framework.

---

## ⚡ Optimized Attention Mechanisms (`ATTENTION_REGISTRY`)

The toolkit supports 70+ attention variants, categorized by their architectural approach and optimization target.

### 1. State-of-the-Art Compressions (Aegis, Nova, Aurora)

- **Aegis (`aegis_v1` to `aegis_v5`, `aegis_hybrid`)**: Advanced security-inspired attention with multi-layer residual compression.
- **Nova (`nova`, `minimal_nova`)**: Features 8 state-of-the-art compression techniques including ARMOR and AST.
- **Aurora (`aurora`, `minimal_aurora`)**: High-performance attention with adaptive spectral filtering.
- **Prism (`prism_full`, `minimal_prism`)**: Multi-spectral attention with hierarchical rank reduction.

### 2. Linear & Low-Rank Approximations

- **Linformer (`linformer`)**: Projects key and value to fixed-size sequence length for $O(N)$ efficiency.
- **SVD Attention (`svd`, `minimal_svd`, `asvd`, `primal`)**: Uses Singular Value Decomposition to approximate the attention matrix.
- **Nystrom (`nystrom`)**: Nystrom-based approximation for linear time complexity.
- **Performer (`performer`)**: Fast Attention via Positive Orthogonal Random Features (FAVOR+).

### 3. Sparse & Block-Based

- **Moba (`moba`)**: Memory-efficient block-based attention.
- **Cluster & LSH (`cluster`, `lsh`)**: Clustering and Locality Sensitive Hashing to find relevant keys.
- **NSA / Adaptive Span (`nsa`, `adaptive_span`)**: Learnable attention spans and sparse access patterns.

### 4. Efficient Architecture Variants

- **GQA / MLA / WGQA (`gqa`, `mla`, `wgqa`)**: Grouped and Multi-Head Latency-Aware attention mechanisms used in modern LLMs (e.g., Llama 3, DeepSeek).
- **Hymba / Jamba (`hymba_transx`, `jamba2`)**: Hybrid attention-mamba architectures for long-context efficiency.
- **Flash3 (`flash3`, `flash3_fp8`)**: Integration with FlashAttention-3 for hardware-accelerated computation.

---

## 📉 Model Compression Techniques (`COMPRESSOR_REGISTRY`)

Our compressors are highly optimized for weight reduction, KV-cache efficiency, and post-training optimization.

### 1. SVD & Matrix Decompositions

- **Swift-SVD (`swift_svd`, `swift_svd_act`)**: Fast, activation-aware Singular Value Decomposition.
- **Advanced SVD (`adasvd`, `aasvd`, `saes_svd_v2`)**: Adaptive and automated SVD for varied architecture layers.
- **CUR / NMF / RPCA (`cur`, `nmf`, `rpca`)**: Alternative matrix factorization methods for high-fidelity compression.

### 2. Pruning & Sparsity

- **SparseGPT / Flap (`sparsegpt`, `flap`)**: One-shot pruning techniques for massive models.
- **ARMOR / AST Pruners (`armor_pruner`, `ast_pruner`)**: NOVA-integrated pruners for rank-modulated reduction.
- **Wanda / SliceGPT (`slicegpt`, `slicegpt_pca`)**: Weight pruning without retraining.

### 3. Quantization & Hashing

- **SAlR / SAES (`salr`, `saes`)**: Advanced quantization-aware reduction.
- **HashedNet / Roast (`hashed_net`, `roast`)**: Hashing-based weight sharing for ultra-low footprint.
- **K-Means / FAQ (`kmeans`, `faq`)**: Codebook-based weight quantization.

### 4. KV-Cache Optimization

- **ZeroKV / HybridKV (`zerokv_neo`, `hybridkv`)**: Techniques to compress the attention KV cache during inference.
- **Delta-KV / CacheNotes (`delta_kv`, `cachenotes`)**: Differential cache compression.

---

## 🛠 Usage & Optimization Tips

- **Selecting a Preset**: Use the `presets/` configuration to quickly choose between `fast_linear` (Linformer/Performer) or `high_accuracy` (NOVA/Aegis).
- **Inference Optimization**: For hardware-constrained environments, `minimal_svd` combined with `kmeans` compression offers the best latency-to-accuracy ratio.
- **Training**: Always use `CompressionTrainer` or `UnslothTrainer` when working with Aegis/Nova mechanisms to ensure correct gradient flow through compressed layers.
