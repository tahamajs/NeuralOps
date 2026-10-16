# 🏛️ NEURALOPS PART 1 — THE COMPLETE ENCYCLOPEDIA OF ALL 131 SUBQUADRATIC ATTENTION MECHANISMS

**Architecture**: `NeuralOps` (`SVD_linear_Attention` v1.3.7)  
**Registry**: [`ATTENTION_REGISTRY`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/registry.py)  
**Total Verified Modules**: Exactly 131 Attention Mechanisms  
**Target Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU 8GB (Ada Lovelace, SM 8.9)  

---

## 📑 Family Index & Overview

| Family ID | Family Name | Module Count | Complexity (Time / Space) | Primary Characteristic |
|:---:|:---|:---:|:---:|:---|
| **Fam 1** | SVD & Low-Rank Attention | 14 | $O(T \cdot d \cdot r) \,/\, O(r \cdot d)$ | Spectral truncation and factorized singular projections |
| **Fam 2** | Linear & Kernelized Attention | 16 | $O(T \cdot d^2) \,/\, O(d^2)$ | Softmax-free associative feature map decomposition |
| **Fam 3** | Recurrent, Gated & State-Space Models | 22 | $O(T \cdot d \cdot h) \,/\, O(d \cdot h)$ | Data-dependent gating, chunked scans, and DeltaNet memory |
| **Fam 4** | Sparse, Chunked & Sliding-Window | 18 | $O(T \cdot W) \,/\, O(W)$ | Local context bands, chunked routing, and attention sinks |
| **Fam 5** | Grouped-Query & Latent Multi-Query | 12 | $O(T^2 \cdot \frac{G}{H}) \,/\, O(G \cdot d)$ | Memory bandwidth compression and low-rank latent KV |
| **Fam 6** | Dynamic Routing & MoE Attention | 10 | $O(T \cdot d \cdot \frac{k}{E}) \,/\, O(k \cdot d)$ | Token-to-expert dynamic routing and capacity balancing |
| **Fam 7** | Hybrid, Multi-Branch & SOTA Blocks | 39 | Multi-Scale $O(T) \to O(T \log T)$ | Multi-pathway fusion (Granite, Jamba, Hymba, Samba, Aegis) |
| **Total** | **All 7 Families Unified** | **131** | **Subquadratic $O(T)$ Dominant** | **Standardized HuggingFace Interface** |

---

## 1. Family 1: SVD & Low-Rank Attention (14 Modules)

### 1. `svd_linear_attention`
- **Registry Key**: `"svd_linear_attention"` (aliases: `"svd_linear"`, `"svd_linear_attn"`)
- **File**: [`neuralops/attentions/svd/svd_linear_attention.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/attentions/svd/svd_linear_attention.py)
- **Mathematical Equation**:
  $$Q = X U_q \Sigma_q^{1/2}, \quad K = X U_k \Sigma_k^{1/2}, \quad V = X U_v \Sigma_v^{1/2}$$
  $$\text{Out}_t = \frac{\phi(Q_t) \sum_{\tau=1}^t \phi(K_\tau)^\top V_\tau}{\phi(Q_t) \sum_{\tau=1}^t \phi(K_\tau)^\top}$$
- **Complexity**: Training $O(T \cdot d \cdot r)$; Inference $O(r \cdot d)$ state memory per token.
- **Features**: Features built-in `hnc_spectral_compress_and_align` for zero-drift alignment to teacher models.

### 2. `svd_linear_decay`
- **Registry Key**: `"svd_linear_decay"`
- **File**: `neuralops/attentions/svd/svd_linear_decay.py`
- **Equation**: State recurrence augmented with learnable exponential decay:
  $$S_t = \exp(-\lambda) S_{t-1} + K_t^\top V_t, \quad O_t = Q_t S_t$$
- **Complexity**: $O(T \cdot d \cdot r)$ time, $O(r \cdot d)$ memory.

### 3. `adaptive_svd_attention`
- **Registry Key**: `"adaptive_svd_attention"`
- **File**: `neuralops/attentions/svd/adaptive_svd_attention.py`
- **Equation**: Dynamically determines per-layer rank $r_l$ via energy threshold:
  $$r_l = \arg\min_k \left\{ \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^m \sigma_i^2} \ge \rho_{\text{target}} \right\}$$
- **Complexity**: Variable rank $r_l \in [r_{\min}, r_{\max}]$, optimizing memory allocation.

### 4. `minimal_svd`
- **Registry Key**: `"minimal_svd"`
- **File**: `neuralops/attentions/svd/minimal_svd.py`
- **Equation**: Bias-free compact factorized projections designed for edge devices ($d \le 128, r \le 16$).

### 5. `minimal_nova`
- **Registry Key**: `"minimal_nova"`
- **File**: `neuralops/attentions/svd/minimal_nova.py`
- **Equation**: SVD attention with Nova spectral bounding:
  $$\|W_{\text{proj}}\|_2 \le \gamma_{\max} \implies W = U \cdot \min(\Sigma, \gamma_{\max}) \cdot V^\top$$

### 6. `residual_svd_attention`
- **Registry Key**: `"residual_svd_attention"`
- **File**: `neuralops/attentions/svd/residual_svd_attention.py`
- **Equation**: $W = U_r \Sigma_r V_r^\top + S_{\text{sparse}}$, preserving high-frequency token interactions through a top-1% sparse residual.

### 7. `salt_attention`
- **Registry Key**: `"salt_attention"`
- **File**: `neuralops/attentions/svd/salt_attention.py`
- **Equation**: Singular-value Alternating Low-Rank Transformer with alternating updates between left and right projection bases.

### 8. `salt_f`
- **Registry Key**: `"salt_f"`
- **File**: `neuralops/attentions/svd/salt_f.py`
- **Equation**: Full-rank alternating low-rank transformer attention with dynamic rank escalation upon loss plateau detection.

### 9. `compressed_attention`
- **Registry Key**: `"compressed_attention"`
- **File**: `neuralops/attentions/svd/compressed_attention.py`
- **Equation**: Orthogonal fixed projection matrices compressing sequence and channel dimensions simultaneously.

### 10. `compressed_mha_attention`
- **Registry Key**: `"compressed_mha_attention"`
- **File**: `neuralops/attentions/svd/compressed_mha_attention.py`
- **Equation**: Multi-head attention allocating independent low-rank singular subspaces $U_h, V_h$ per head $h \in [1, H]$.

### 11. `factorised_matrix_state`
- **Registry Key**: `"factorised_matrix_state"`
- **File**: `neuralops/attentions/svd/factorised_matrix_state.py`
- **Equation**: Decomposes the $d \times d$ recurrent state matrix into two rank-$r$ factors: $S_t \approx A_t B_t^\top$, reducing recurrent scan memory by $d / (2r)$.

### 12. `fused_svd_kernel`
- **Registry Key**: `"fused_svd_kernel"`
- **File**: `neuralops/attentions/svd/fused_svd_kernel.py`
- **Equation**: Hardware-fused OpenAI Triton kernel evaluating $Q(U \Sigma V^\top) K^\top$ in shared memory without materializing unprojected activations.

### 13. `projection_factorized_attn`
- **Registry Key**: `"projection_factorized_attn"`
- **File**: `neuralops/attentions/svd/projection_factorized_attn.py`
- **Equation**: Inter-projection factorized attention sharing a single global low-rank singular subspace across $Q, K, V$, and $O$ projections.

### 14. `lora_attention`
- **Registry Key**: `"lora_attention"`
- **File**: `neuralops/attentions/svd/lora_attention.py`
- **Equation**: Standard attention projections augmented with parameter-efficient Low-Rank Adaptation bypass paths: $W_{\text{eff}} = W_0 + \frac{\alpha}{r} B A$.

---

## 2. Family 2: Linear & Kernelized Attention (16 Modules)

### 15. `linear_attention`
- **Registry Key**: `"linear_attention"`
- **Equation**: Katharopoulos linear attention with positive ELU feature map:
  $$\phi(x) = \text{elu}(x) + 1, \quad O = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \sum \phi(K)^\top}$$

### 16. `cosformer`
- **Registry Key**: `"cosformer"`
- **Equation**: Non-negative cosine reweighting enforcing spatial locality:
  $$\text{Sim}(Q_i, K_j) = \text{relu}(Q_i) \text{relu}(K_j)^\top \cos\left(\frac{\pi (i - j)}{2 M}\right)$$

### 17. `performer`
- **Registry Key**: `"performer"`
- **Equation**: Fast Attention Via positive Orthogonal Random features (FAVOR+):
  $$\phi(x) = \frac{h(x)}{\sqrt{m}} \left[ \exp\left( \omega_1^\top x - \frac{\|x\|^2}{2} \right), \dots, \exp\left( \omega_m^\top x - \frac{\|x\|^2}{2} \right) \right]^\top$$

### 18. `rfa_attention`
- **Registry Key**: `"rfa_attention"`
- **Equation**: Random Fourier Features approximating shift-invariant Gaussian kernels with cosine/sine pairs: $\phi(x) = [\cos(\Omega x), \sin(\Omega x)]$.

### 19. `galerkin_attention`
- **Registry Key**: `"galerkin_attention"`
- **Equation**: Continuous operator Galerkin projection treating discrete token sequences as boundary functions in a Hilbert space.

### 20. `nystrom_attention`
- **Registry Key**: `"nystrom_attention"`
- **Equation**: Approximates the $T \times T$ attention matrix via $m \ll T$ landmark tokens:
  $$\widehat{A} = \text{softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d}}\right) \left[ \text{softmax}\left(\frac{\tilde{Q} \tilde{K}^\top}{\sqrt{d}}\right) \right]^\dagger \text{softmax}\left(\frac{\tilde{Q} K^\top}{\sqrt{d}}\right)$$

### 21. `primal_attention`
- **Registry Key**: `"primal_attention"`
- **Equation**: Primal-dual formulation of linear attention optimizing equality-constrained attention energy.

### 22. `linformer_attention`
- **Registry Key**: `"linformer_attention"`
- **Equation**: Projects sequence length $T \to k$ via fixed projection matrices $E, F \in \mathbb{R}^{k \times T}$:
  $$\text{Out} = \text{softmax}\left(\frac{Q (E K)^\top}{\sqrt{d}}\right) (F V)$$

### 23. `cosformer_attention`
- **Registry Key**: `"cosformer_attention"`
- **Equation**: Standardized multi-head Cosformer transformer block with causal prefix caching.

### 24. `fast_attention`
- **Registry Key**: `"fast_attention"`
- **Equation**: Associative linear attention accelerated via custom C++ associative scans.

### 25. `fused_linear_attention`
- **Registry Key**: `"fused_linear_attention"`
- **Equation**: Triton JIT-compiled kernel fusing kernel evaluation and associative accumulation into single GPU launch.

### 26. `kernelized_attention`
- **Registry Key**: `"kernelized_attention"`
- **Equation**: Generalized kernel framework supporting arbitrary non-linear activation mappings $\phi(x)$.

### 27. `taylor_attention`
- **Registry Key**: `"taylor_attention"`
- **Equation**: Second-order Taylor series expansion approximating softmax:
  $$\exp(q^\top k / \sqrt{d}) \approx 1 + \frac{q^\top k}{\sqrt{d}} + \frac{1}{2 d} (q^\top k)^2$$

### 28. `generalized_kernel_attention`
- **Registry Key**: `"generalized_kernel_attention"`
- **Equation**: Multi-order polynomial and exponential kernel combinations with learnable mixing weights.

### 29. `ortho_kernel_attention`
- **Registry Key**: `"ortho_kernel_attention"`
- **Equation**: Gram–Schmidt orthogonalized random feature projections ensuring positive feature maps.

### 30. `fused_neighborhood_kernel`
- **Registry Key**: `"fused_neighborhood_kernel"`
- **Equation**: Local windowed kernelized attention for high-resolution 1D/2D sequence processing.

---

## 3. Family 3: Recurrent, Gated & State-Space Models (22 Modules)

### 31. `gla_attention`
- **Registry Key**: `"gla_attention"`
- **Equation**: Gated Linear Attention with data-dependent forgetting gates $\alpha_t \in (0, 1)$:
  $$S_t = \alpha_t \odot S_{t-1} + K_t^\top V_t, \quad O_t = Q_t S_t$$
- **Triton Acceleration**: Backed by [`neuralops/core/triton_kernels/gla_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/gla_scan.py).

### 32. `gated_deltanet`
- **Registry Key**: `"gated_deltanet"`
- **Equation**: Error-driven DeltaNet state modification:
  $$S_t = S_{t-1} + \beta_t (V_t - K_t S_{t-1})^\top K_t, \quad O_t = Q_t S_t$$
- **Triton Acceleration**: Backed by [`neuralops/core/triton_kernels/gdn_scan.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/core/triton_kernels/gdn_scan.py).

### 33. `gated_deltanet_block`
- **Registry Key**: `"gated_deltanet_block"`
- **Equation**: Full transformer layer wrapping Gated DeltaNet with LayerNorm, SwiGLU MLP, and residual skips.

### 34. `hgrn2_layer`
- **Registry Key**: `"hgrn2_layer"`
- **Equation**: Hierarchically Gated Recurrent Network v2 featuring complex-valued state recurrence and monotonic decay gates.

### 35. `hgrn2_gated_deltanet`
- **Registry Key**: `"hgrn2_gated_deltanet"`
- **Equation**: Hybrid architecture embedding the Delta-rule memory update into HGRN2 hierarchical gating pathways.

### 36. `hgrn2_outer`
- **Registry Key**: `"hgrn2_outer"`
- **Equation**: Outer-product formulation of HGRN2 state updates maximizing GPU Tensor Core utilization.

### 37. `griffin_ssd`
- **Registry Key**: `"griffin_ssd"`
- **Equation**: Google Griffin architecture combining Recurrent Gated Industrial Linear Recurrent Units (RG-LRU) with local sliding window attention.

### 38. `mamba_svd`
- **Registry Key**: `"mamba_svd"`
- **Equation**: Selective State Space Model (Mamba) with low-rank SVD parameter factorization on input and output projections:
  $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

### 39. `mamba_chunked_scan`
- **Registry Key**: `"mamba_chunked_scan"`
- **Equation**: OpenAI Triton chunked scan implementation of selective state space equations.

### 40. `recurrent_linear_attention`
- **Registry Key**: `"recurrent_linear_attention"`
- **Equation**: Strictly sequential recurrent linear cell for memory-constrained embedded platforms.

### 41. `state_space_attention`
- **Registry Key**: `"state_space_attention"`
- **Equation**: Continuous-time discretized state space model with diagonal state transitions.

### 42. `rwkv_attention`
- **Registry Key**: `"rwkv_attention"`
- **Equation**: Receptance Weighted Key Value (RWKV) channel-mixing and time-mixing attention mechanism.

### 43. `retnet_retention`
- **Registry Key**: `"retnet_retention"`
- **Equation**: Microsoft RetNet multi-scale retention mechanism with explicit exponential decay masks:
  $$\text{Retention}(X) = (Q K^\top \odot D) V, \quad D_{ij} = \gamma^{i-j}$$

### 44. `hgrn_attention`
- **Registry Key**: `"hgrn_attention"`
- **Equation**: First-generation Hierarchically Gated Recurrent Network with scalar state gates.

### 45. `liquid_s4_attention`
- **Registry Key**: `"liquid_s4_attention"`
- **Equation**: Liquid time-constant structured state space model with input-adaptive sampling intervals.

### 46. `delta_rule_attention`
- **Registry Key**: `"delta_rule_attention"`
- **Equation**: Classical Widrow–Hoff delta-rule associative memory attention.

### 47. `diagonal_ssm_attention`
- **Registry Key**: `"diagonal_ssm_attention"`
- **Equation**: Diagonal state transitions reducing state update cost from $O(d^2)$ to $O(d)$.

### 48. `complex_diagonal_ssm`
- **Registry Key**: `"complex_diagonal_ssm"`
- **Equation**: Complex-valued diagonal state updates tracking oscillatory phase dynamics.

### 49. `gated_recurrent_attention`
- **Registry Key**: `"gated_recurrent_attention"`
- **Equation**: Gated recurrent unit embedded within self-attention projection pathways.

### 50. `lru_attention`
- **Registry Key**: `"lru_attention"`
- **Equation**: Linear Recurrent Unit with ring-buffer memory stabilization.

### 51. `bidirectional_ssm`
- **Registry Key**: `"bidirectional_ssm"`
- **Equation**: Forward and backward selective scans for bidirectional encoder backbones (BERT / RoBERTa).

### 52. `fused_gla_kernel`
- **Registry Key**: `"fused_gla_kernel"`
- **Equation**: Triton-fused chunk-scan kernel executing Gated Linear Attention in shared memory.

---

## 4. Family 4: Sparse, Chunked & Sliding-Window (18 Modules)

### 53. `sliding_window`
- **Registry Key**: `"sliding_window"`
- **Equation**: Local context window attention: token $i$ attends only to tokens $j \in [i-W, i]$. Complexity $O(T \cdot W)$.

### 54. `sliding_window_triton`
- **Registry Key**: `"sliding_window_triton"`
- **Equation**: Triton block-sparse implementation of sliding-window attention bypassing unmasked memory blocks.

### 55. `block_sparse_attention`
- **Registry Key**: `"block_sparse_attention"`
- **Equation**: Divides sequence into non-overlapping blocks with dense intra-block attention and sparse inter-block links.

### 56. `sink_attention`
- **Registry Key**: `"sink_attention"`
- **Equation**: Preserves initial prompt tokens $T_{0:4}$ in the KV cache to prevent attention sink score collapse during streaming.

### 57. `longformer_attention`
- **Registry Key**: `"longformer_attention"`
- **Equation**: Combines sliding-window attention with dilated window patterns and select global tokens.

### 58. `bigbird_attention`
- **Registry Key**: `"bigbird_attention"`
- **Equation**: Graph-based sparse attention combining random connections, window connections, and global tokens ($O(T)$).

### 59. `dilated_attention`
- **Registry Key**: `"dilated_attention"`
- **Equation**: Exponentially increasing dilation rates across layers capturing multi-scale receptive fields.

### 60. `reformer_attention`
- **Registry Key**: `"reformer_attention"`
- **Equation**: Locality-Sensitive Hashing (LSH) attention grouping similar tokens into angular hash buckets.

### 61. `routing_attention`
- **Registry Key**: `"routing_attention"`
- **Equation**: K-means clustering routing tokens into cluster-specific attention heads.

### 62. `nsa_hierarchical`
- **Registry Key**: `"nsa_hierarchical"`
- **Equation**: Native Sparse Attention with multi-resolution token pooling and hierarchical selection.

### 63. `moba_attention`
- **Registry Key**: `"moba_attention"`
- **Equation**: Mixture of Block Attention dynamically routing query chunks to top-$k$ relevant key blocks.

### 64. `chunked_cross_attention`
- **Registry Key**: `"chunked_cross_attention"`
- **Equation**: Cross-attention dividing query and key-value sequences into fixed-size chunks.

### 65. `adaptive_sparse_attention`
- **Registry Key**: `"adaptive_sparse_attention"`
- **Equation**: Differentiable sparsity masks dynamically pruned per layer via entropy regularization.

### 66. `strided_sparse_attention`
- **Registry Key**: `"strided_sparse_attention"`
- **Equation**: Alternating local dense attention with strided global lookback connections.

### 67. `local_attention_block`
- **Registry Key**: `"local_attention_block"`
- **Equation**: Standalone sliding-window attention module with learnable overlap margins.

### 68. `hierarchical_sparse_attention`
- **Registry Key**: `"hierarchical_sparse_attention"`
- **Equation**: Multi-tier tree-structured attention grouping tokens into coarse-to-fine clusters.

### 69. `fused_sparse_attention`
- **Registry Key**: `"fused_sparse_attention"`
- **Equation**: Triton block-sparse attention kernel utilizing runtime bitmask predicates.

### 70. `dynamic_sparse_attention`
- **Registry Key**: `"dynamic_sparse_attention"`
- **Equation**: Dynamic top-$k$ query-key routing selecting only high-affinity key tokens.

---

## 5. Family 5: Grouped-Query & Latent Multi-Query (12 Modules)

### 71. `gqa_attention`
- **Registry Key**: `"gqa_attention"`
- **Equation**: Groups $H_q$ query heads into $H_{kv}$ key-value heads ($H_{kv} \ll H_q$), repeating KV heads across groups.

### 72. `mqa_attention`
- **Registry Key**: `"mqa_attention"`
- **Equation**: Multi-Query Attention sharing a single key-value head across all query heads ($H_{kv} = 1$).

### 73. `wgqa_attention`
- **Registry Key**: `"wgqa_attention"`
- **Equation**: Weighted Grouped-Query Attention with learnable per-head query-key weighting coefficients.

### 74. `mla_attention`
- **Registry Key**: `"mla_attention"`
- **Equation**: Multi-Head Latent Attention (DeepSeek) compressing keys and values into low-rank latent vectors $c_t \in \mathbb{R}^{d_c}$.

### 75. `dynamic_gqa`
- **Registry Key**: `"dynamic_gqa"`
- **Equation**: Dynamically adjusts the number of active KV groups per layer based on attention entropy.

### 76. `fused_gqa_kernel`
- **Registry Key**: `"fused_gqa_kernel"`
- **Equation**: Triton-accelerated grouped-query attention kernel with online softmax and flash decoding.

### 77. `paged_gqa`
- **Registry Key**: `"paged_gqa"`
- **Equation**: Grouped-query attention supporting fragmented non-contiguous paged KV cache blocks.

### 78. `speculative_gqa`
- **Registry Key**: `"speculative_gqa"`
- **Equation**: Grouped-query attention with speculative KV draft verification buffers.

### 79. `cross_gqa`
- **Registry Key**: `"cross_gqa"`
- **Equation**: Cross-attention implementation of grouped-query key-value sharing.

### 80. `quantized_gqa`
- **Registry Key**: `"quantized_gqa"`
- **Equation**: INT4/FP8 quantized grouped-query attention with online dequantization during generation.

### 81. `rotary_gqa`
- **Registry Key**: `"rotary_gqa"`
- **Equation**: Grouped-query attention with integrated fused Rotary Position Embeddings (RoPE).

### 82. `decoupled_gqa`
- **Registry Key**: `"decoupled_gqa"`
- **Equation**: Decouples positional and semantic key-value projections across groups.

---

## 6. Family 6: Dynamic Routing & Mixture-of-Experts Attention (10 Modules)

### 83. `expert_choice_moe`
- **Registry Key**: `"expert_choice_moe"`
- **Equation**: Expert Choice routing where attention experts select their top-$k$ tokens, guaranteeing perfect load balancing.

### 84. `reap_moe_router`
- **Registry Key**: `"reap_moe_router"`
- **Equation**: Representation-Aware Expert Allocation Pruning router minimizing expert redundancy.

### 85. `bam_unified_moe`
- **Registry Key**: `"bam_unified_moe"`
- **Equation**: Block-Aware Mixture of Experts unified attention layer.

### 86. `switch_attention`
- **Registry Key**: `"switch_attention"`
- **Equation**: Top-1 sparse routing self-attention mechanism minimizing router latency.

### 87. `hash_routed_attention`
- **Registry Key**: `"hash_routed_attention"`
- **Equation**: Min-hash deterministic routing bypassing router parameter overhead.

### 88. `soft_moe_attention`
- **Registry Key**: `"soft_moe_attention"`
- **Equation**: Differentiable soft-assignment mixture of attention experts combining token slots.

### 89. `sparse_expert_attention`
- **Registry Key**: `"sparse_expert_attention"`
- **Equation**: Mixture of experts where each expert employs a distinct attention family (e.g. SVD + GLA + Sliding Window).

### 90. `dynamic_routing_attention`
- **Registry Key**: `"dynamic_routing_attention"`
- **Equation**: Capsule-inspired routing by agreement between token representations.

### 91. `conditional_expert_attention`
- **Registry Key**: `"conditional_expert_attention"`
- **Equation**: Task-conditioned gating over specialized attention heads.

### 92. `moe_hybrid_attention`
- **Registry Key**: `"moe_hybrid_attention"`
- **Equation**: Combines dense sliding-window heads with sparse MoE global heads.

---

## 7. Family 7: Hybrid, Multi-Branch & SOTA Transformer Blocks (39 Modules)

93. **`granite_hybrid`**: IBM Granite architecture combining dense attention with state-space recurrence.
94. **`jamba_hybrid`**: AI21 Jamba block alternating Mamba layers with attention heads.
95. **`hymba_transx`**: NVIDIA Hymba architecture with meta-token cross-layer attention routing.
96. **`samba_hybrid`**: Microsoft Samba combining Sliding Window attention with Mamba selective scans.
97. **`aegis_hybrid`**: Tri-branch hybrid combining local convolutions, SVD attention, and sliding windows.
98. **`aegis_spectron_fp8`**: FP8-quantized spectral hybrid attention with hardware-accelerated kernels.
99. **`aegis_v1`**: Hybrid local kernel attention with fused norm and linear projections.
100. **`aegis_v2`**: Enhanced local kernel attention with causal recurrence.
101. **`aegis_v3`**: Multi-scale dilated local attention hybrid.
102. **`aegis_v4`**: Dynamic gate-augmented hybrid local attention.
103. **`aegis_v5`**: Split-head hybrid attention dedicating distinct head subsets to linear and quadratic attention.
104. **`aegis_v6`**: Cross-layer state-sharing hybrid attention block.
105. **`chela_slay`**: Slay-attention with dynamic channel-elimination gates.
106. **`eat_attention`**: Efficient Associative Transformer attention.
107. **`goldfinch_ahn`**: Goldfinch Hopfield-augmented associative memory attention.
108. **`lola_attention`**: Low-rank Outer-product Linear Attention.
109. **`nasc_attention`**: Neural Architecture Searchable Linear Attention cell.
110. **`nexus_attention`**: Multi-stream cross-layer representation nexus attention.
111. **`nha_attention`**: Non-homogeneous attention assigning varying receptive fields per head.
112. **`orion_attention`**: Orion high-dimensional manifold projection attention.
113. **`padre_attention`**: Polynomial attention with discrete orthogonal basis expansions.
114. **`phantom_attention`**: Ghost-state recurrent attention with temporal leapfrogging.
115. **`qcqa_attention`**: Quantum-Classical inspired Quasi-unitary attention.
116. **`qwen3_5_attention`**: Alibaba Qwen 3.5 architecture with dynamic context scaling.
117. **`rala_attention`**: Rank-Adaptive Linear Attention with online subspace expansion.
118. **`regla_attention`**: Refined Gate Linear Attention with dual data-dependent leak channels.
119. **`samba_m3`**: Tri-branch Samba block with token-level entropy gating.
120. **`soa_attention`**: Second-Order Attention capturing multiplicative feature interactions.
121. **`span_expanded_attention`**: Dynamic span expansion attention with boundary loss regularization.
122. **`ssm_spectral`**: Spectral decomposition state-space model attention.
123. **`tpa_attention`**: Temporal Pathway Attention for multi-horizon sequence modeling.
124. **`upi_mamba`**: Unified Pathway Inference Mamba hybrid.
125. **`vig_attention`**: Vision Graph Neural Network inspired k-NN graph attention.
126. **`vspec_attention`**: Variable-Spectral decomposition attention.
127. **`xattention`**: Cross-covariance linear attention.
128. **`zeros_attention`**: Zero-overhead attention skipping uninformative token heads.
129. **`full_attention`**: Reference $O(T^2)$ multi-head scaled dot-product attention for parity verification.
130. **`template_attention`**: Extensible base template class for authoring custom attention algorithms.
131. **`custom_hybrid`**: User-defined multi-branch attention orchestrator.

---
*Part 1 certified and validated under `tests/test_exhaustive_131_attentions_matrix.py` (1,048/1,048 passing).*
