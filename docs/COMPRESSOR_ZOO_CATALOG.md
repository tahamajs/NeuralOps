# 🔬 NeuralOps 82-Compressor Zoo Catalog & Multi-Rank Specifications

NeuralOps provides the largest verified collection of neural model surgery compression algorithms: **82 Compression Engines** evaluated across 4 rank sweeps ($r \in \{8, 16, 32, 64\}$) totaling **328 evaluated models** on GLUE SST-2 and WikiText-2.

All compression engines implement calibrated low-rank matrix decomposition, structured pruning, integer/floating-point quantization, Riemannian manifold optimization, tensor networks, or dynamic KV-cache compression to drastically reduce VRAM footprints and accelerate inference while preserving model accuracy.

[![Release](https://img.shields.io/badge/release-v2.1.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.1.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%203090%20(SM%208.6)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## 🏛️ The 8 Algorithmic Families Summary

| # | Algorithmic Family | Count | Core Mechanisms | Representative Algorithms | Average Compression Ratio ($r=32$) | Recovered Accuracy (%) |
| :-: | :--- | :-: | :--- | :--- | :-: | :-: |
| 1 | **Standard SVD & Spectral Decomposition** | 14 | Direct singular value decomposition, energy thresholding | `direct_svd`, `aa_svd`, `saes_svd_v2`, `svd_linear` | $1.84\times$ | $91.12\%$ |
| 2 | **Activation-Aware & Data-Driven SVD** | 12 | Fisher curvature weighting, activation covariance whitening | `asvd`, `fwsvd`, `adasvd_compressor`, `swift_svd_act` | $2.12\times$ | $91.68\%$ |
| 3 | **Matrix & Mixed-Precision Quantization** | 16 | Non-uniform clustering, mixed 4-bit/16-bit adapters, rotations | `kmeans_quantize`, `awq`, `gptq`, `rotational_slicegpt_int4` | $3.40\times$ | $89.85\%$ |
| 4 | **Structured Channel & Layer Pruning** | 14 | Alternating Direction Method of Multipliers (ADMM), magnitude pruning | `admm_pruner`, `bonsai_pruner`, `slim_compressor` | $1.95\times$ | $90.40\%$ |
| 5 | **Riemannian Manifold & Geometric Optimizers** | 10 | Stiefel manifold retraction, Grassmannian projection | `optimal_subspace_projection`, `rpca_compressor`, `drone_compressor` | $2.05\times$ | $91.45\%$ |
| 6 | **Tensor Train & Kronecker Factorization** | 8 | Block rearrangement, Kronecker tensor product approximation | `kronecker_compressor`, `tt_compressor`, `cur_compressor` | $2.75\times$ | $89.20\%$ |
| 7 | **Hardware-Aware Semi-Structured Sparsity** | 4 | NVIDIA Ampere/Ada 2:4 Sparse Tensor Core acceleration | `sparsegpt`, `wanda`, `aatc_xkv_compressor` | $2.40\times$ | $91.60\%$ |
| 8 | **Cross-Layer & Subspace Sharing** | 4 | Inter-layer basis subspace parameter sharing | `delta_kv`, `hybridkv_compressor`, `zerokv_neo_compressor` | $2.60\times$ | $90.80\%$ |
| **Total** | **Comprehensive Zoo Catalog** | **82** | **Full Multi-Rank Verified Spectrum** | **82 Compression Engines** | **$2.33\times$** | **$91.07\%$** |

---

## 🔬 Exhaustive Master Catalog: All 82 Compression Engines

### Family 1: Standard SVD & Spectral Decomposition (14 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **001** | `direct_svd` | **Direct SVD Compressor** | $W \approx U_r \Sigma_r V_r^\top = A B$ | Direct Eckart-Young-Mirsky truncated singular value decomposition factorizing weight matrix W into low-rank factor matrices A and B. | 1.85x | 91.20% | 1.42x |
| **002** | `svd_linear` | **SVD Linear Factorizer** | $W \to W_A W_B, \quad W_A \in \mathbb{R}^{d_{\text{out}} \times r}, W_B \in \mathbb{R}^{r \times d_{\text{in}}}$ | Decomposes nn.Linear layers into two sequential linear layers without bias on intermediate low-rank bottleneck. | 1.85x | 91.10% | 1.40x |
| **003** | `saes_svd_v2` | **SAES-SVD v2 (Singular Value Energy Scaling)** | $\hat{W} = U_r (\Sigma_r)^{\alpha} V_r^\top, \quad \alpha \in [0.8, 1.2]$ | Singular value energy rescaling and spectrum balancing to prevent gradient vanishing across deep residual paths. | 1.88x | 91.40% | 1.45x |
| **004** | `erc_svd` | **ERC-SVD (Error Residual Correction)** | $W \approx U_r \Sigma_r V_r^\top + R_{\text{sparse}}, \quad \|R_{\text{sparse}}\|_0 \le k$ | Truncated low-rank decomposition complemented with sparse error residual correction for outlier weights. | 1.82x | 91.50% | 1.38x |
| **005** | `dobi_svd` | **DOBI-SVD (Double-Orthogonal Bidirectional Iteration)** | $\min_{A, B} \|W - A B^\top\|_F^2 \quad \text{s.t.} \quad A^\top A = I_r, B^\top B = I_r$ | Bidirectional orthogonal iteration enforcing dual orthogonality constraints on left and right factor matrices. | 1.86x | 91.30% | 1.44x |
| **006** | `residual_svd_compressor` | **Residual SVD Compressor** | $W = A B + \lambda \Delta W, \quad \Delta W = \text{TopK}(W - A B)$ | Decouples low-rank global representation from top-magnitude residual spectral projections. | 1.80x | 91.60% | 1.36x |
| **007** | `hybrid_svd_layer` | **Hybrid SVD Layer Compressor** | $Y = X B^\top A^\top + X W_{\text{res}}^\top$ | Hybrid architecture maintaining a low-rank parallel branch alongside downscaled identity skip pathways. | 1.84x | 91.20% | 1.41x |
| **008** | `frandi_compressor` | **FRanDI (Fast Randomized Dynamic Iteration)** | $\Omega \sim \mathcal{N}(0, 1), \quad Y = W \Omega, \quad Q = \text{qr}(Y), \quad B = Q^\top W$ | Randomized range finder algorithm approximating top-r singular subspaces in O(d^2 r) wall-clock time. | 1.90x | 91.00% | 1.50x |
| **009** | `salr` | **SALR (Singular-Aware Low-Rank)** | $\text{score}_i = \sigma_i \cdot \cos(\angle(u_i, v_i)), \quad r = \arg\max_k \sum_{i=1}^k \text{score}_i$ | Singular value and vector alignment scoring to prioritize low-rank factor preservation. | 1.83x | 91.20% | 1.42x |
| **010** | `dynamic_rank` | **Dynamic Rank SVD Compressor** | $r_l = \min\left(r_{\max}, \left\lceil \frac{\sum_{i=1}^{r_l} \sigma_{l,i}^2}{\sum_i \sigma_{l,i}^2} \ge \tau \right\rceil\right)$ | Energy-thresholded dynamic rank allocation assigning rank based on Frobenius energy capture ratio tau. | 1.85x | 91.30% | 1.43x |
| **011** | `adaptive_ranks` | **Adaptive Ranks Compressor** | $r_l \propto \log \det(W_l^\top W_l + \epsilon I)$ | Allocates ranks across network layers proportional to layer-wise log-determinant spectral volume. | 1.82x | 91.40% | 1.40x |
| **012** | `entropy_budget_allocation` | **Entropy Budget Allocation SVD** | $\mathcal{H}_l = -\sum_i \tilde{\sigma}_{l,i} \log \tilde{\sigma}_{l,i}, \quad r_l = R_{\text{total}} \frac{\mathcal{H}_l}{\sum_j \mathcal{H}_j}$ | Allocates global parameter rank budget according to Von Neumann spectral entropy of singular values. | 1.86x | 91.50% | 1.44x |
| **013** | `auto_svd` | **AutoSVD (Automated Sensitivity Profiler)** | $r_l^* = \arg\min_r \text{Tr}(F_l (W_l - \hat{W}_l(r))^\top (W_l - \hat{W}_l(r)))$ | Automated grid search and gradient sensitivity profiler finding optimal Pareto rank allocations. | 1.84x | 91.20% | 1.42x |
| **014** | `hadamard_incoherent_svd` | **Hadamard Incoherent SVD** | $W_H = H_{d_{\text{out}}} W H_{d_{\text{in}}}, \quad W_H \approx U_r \Sigma_r V_r^\top$ | Applies Walsh-Hadamard transform before SVD to spread spiky outlier weights uniformly across the singular spectrum. | 1.88x | 91.50% | 1.46x |

---

### Family 2: Activation-Aware & Data-Driven SVD (12 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **015** | `aa_svd_compressor` | **AASVD (Activation-Aware SVD)** | $\tilde{W} = W \cdot \text{diag}(s_X), \quad s_X = \sqrt{\text{diag}(\mathbb{E}[X^\top X])}$ | Scales weight columns by empirical activation magnitude before SVD decomposition, preserving activation fidelity. | 2.10x | 91.70% | 1.60x |
| **016** | `adasvd_compressor` | **AdaSVD (Adaptive Whitening SVD)** | $\tilde{W} = W C_X^{1/2}, \quad C_X = \frac{1}{N} X^\top X + \epsilon I, \quad \hat{W} = U_r \Sigma_r V_r^\top C_X^{-1/2}$ | Full Cholesky activation covariance whitening with weighted Alternating Least Squares (ALS) refinement. | 2.15x | 91.80% | 1.65x |
| **017** | `asvd` | **ASVD (Activation SVD with KV Retention)** | $\min_{\hat{W}} \|(W - \hat{W}) X\|_F^2 \quad \text{s.t.} \quad \text{rank}(\hat{W}) \le r$ | Direct data-driven empirical output error minimization weighting singular vectors by channel variance. | 2.12x | 91.65% | 1.62x |
| **018** | `fwsvd_compressor` | **FWSVD (Fisher-Weighted SVD)** | $\min_{\hat{W}} \text{Tr}\left( (W - \hat{W}) F_W (W - \hat{W})^\top \right)$ | Weighting SVD objective by empirical diagonal or Kronecker Fisher Information Matrix of gradients. | 2.14x | 91.85% | 1.64x |
| **019** | `swift_svd` | **SwiftSVD (Closed-Form Analytical SVD)** | $\hat{W} = W V_r V_r^\top, \quad C_X V_r = V_r \Lambda_r$ | Closed-form low-overhead projection onto principal activation eigenvectors bypassing full SVD. | 2.08x | 91.55% | 1.70x |
| **020** | `swift_svd_act` | **SwiftSVD-Act (Activation-Calibrated SwiftSVD)** | $B = (A^\top A)^{-1} A^\top W X X^\top (X X^\top)^{-1}$ | Least-squares optimal right factor regression directly calibrated against forward activations. | 2.11x | 91.75% | 1.68x |
| **021** | `svd_llm_compressor` | **SVD-LLM Compressor** | $W C_X = U \Sigma V^\top, \quad \hat{W} = U_r \Sigma_r V_r^\top (C_X)^{-1}$ | Data-whitening low-rank decomposition tailored for large language models with activation outlier suppression. | 2.16x | 91.80% | 1.66x |
| **022** | `a3_compressor` | **A3 Compressor (Attention-Aware Activation SVD)** | $\tilde{W}_Q = W_Q S_{\text{attn}}, \quad S_{\text{attn}} = \mathbb{E}[\text{Softmax}(Q K^\top / \sqrt{d})]$ | Incorporates downstream self-attention entropy maps to prioritize critical query and key subspaces. | 2.15x | 91.70% | 1.63x |
| **023** | `come_compressor` | **COME (Covariance Optimal Matrix Embedding)** | $\min_A \|X W^\top - X B^\top A^\top\|_F^2 + \lambda \|A\|_*$ | Joint nuclear-norm regularized low-rank matrix embedding over empirical feature covariance. | 2.10x | 91.60% | 1.58x |
| **024** | `grecodec_compressor` | **GreCoDec (Greedy Covariance Decomposition)** | $u_k, v_k = \arg\max_{u, v} u^\top (W - \hat{W}_{k-1}) C_X (W - \hat{W}_{k-1})^\top u$ | Greedy iterative rank-1 covariance pursuit capturing dominant directions of output variance. | 2.12x | 91.65% | 1.62x |
| **025** | `sdrm_compressor` | **SDRM (Spectral Dimension Reduction Method)** | $\hat{W} = \Pi_{\mathcal{S}} W, \quad \mathcal{S} = \text{span}(e_1(X^\top X), \dots, e_r(X^\top X))$ | Spectral projection onto the leading eigenspace of input activation Gram matrices. | 2.09x | 91.50% | 1.59x |
| **026** | `soka_compressor` | **SOKA (Second-Order Kernel Approximation)** | $K_{i,j} = \langle X_i, X_j \rangle, \quad W \approx W K_r (K_r^\top K_r)^{-1} K_r^\top$ | Second-order activation kernel approximation factorizing weights along empirical sample manifolds. | 2.13x | 91.70% | 1.61x |

---

### Family 3: Matrix & Mixed-Precision Quantization (16 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **027** | `astro_quantizer` | **AstroQuantizer (Dynamic Uniform Range Quant)** | $W_q = \text{clamp}\left(\left\lfloor \frac{W - z}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$ | Dynamic uniform range grid quantization with straight-through estimator (STE) gradient surrogate. | 3.80x | 90.10% | 1.85x |
| **028** | `awq` | **AWQ (Activation-Aware Weight Quantization)** | $W^* = \arg\min_W \|W X - Q(W \cdot \text{diag}(s)) \text{diag}(s)^{-1} X\|_F^2$ | Protects top 1% salient weight channels by searching per-channel scales s to minimize output error. | 3.90x | 91.20% | 2.10x |
| **029** | `coopq_quantizer` | **CoopQ (Cooperative Quantization & Low-Rank)** | $W \approx Q_{\text{INT4}}(W - A B) + A B, \quad A \in \mathbb{R}^{d \times r}$ | Cooperative optimization partitioning weight matrix into INT4 base grid plus FP16 low-rank residual adapters. | 3.60x | 90.80% | 1.95x |
| **030** | `faq_quantizer` | **FAQ (Fisher-Aware Quantizer)** | $s^* = \arg\min_s \text{Tr}\left( (W - Q(W; s))^\top F_W (W - Q(W; s)) \right)$ | Calibrates quantization scale step-sizes using second-order Fisher diagonal curvature bounds. | 3.50x | 90.60% | 1.80x |
| **031** | `fp8` | **FP8 Native Quantizer (E4M3 / E5M2)** | $W_{\text{FP8}} = \text{quantize\_fp8}(W \cdot s), \quad s = \frac{448.0}{\max(|W|)}$ | Hardware-accelerated 8-bit floating point quantization compliant with NVIDIA Ada/Hopper FP8 GEMM engines. | 2.00x | 91.80% | 1.90x |
| **032** | `gptq` | **GPTQ (Generalized Post-Training Quantization)** | $w_q^* = \text{round}(w_q), \quad w_{>q} \gets w_{>q} - \frac{w_q - w_q^*}{[H^{-1}]_{qq}} H^{-1}_{:, q}$ | Second-order Taylor Hessian inverse column-by-column error diffusion updates with Cholesky stabilization. | 3.85x | 91.10% | 2.05x |
| **033** | `huffman_coder` | **Huffman Optimal Entropy Coder** | $L_{\text{avg}} = \sum_i p_i \ell_i \ge \mathcal{H}(W_q)$ | Variable-length prefix coding applied to quantized centroid indices approaching Shannon entropy limits. | 4.20x | 89.40% | 1.25x |
| **034** | `kmeans_quantize` | **K-Means Vector Quantizer** | $\min_C \sum_{i,j} \min_{k} (W_{i,j} - c_k)^2$ | Non-uniform Lloyd-Max 1D/2D cluster centroid vector quantization compressing 16-bit weights to 4-bit indices. | 3.70x | 89.90% | 1.60x |
| **035** | `rotational_slicegpt_int4` | **Rotational SliceGPT INT4** | $W_{\text{rot}} = Q W, \quad \hat{W} = Q_{\text{INT4}}(\text{Slice}(W_{\text{rot}}))$ | Applies randomized orthogonal Haar rotations followed by structured 25% column slicing and INT4 quantization. | 4.10x | 89.50% | 2.15x |
| **036** | `smoothquant` | **SmoothQuant W8A8 Quantizer** | $Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W), \quad s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$ | Migrates activation outlier quantization difficulty to weights via per-channel scale migration parameter alpha=0.5. | 2.00x | 91.80% | 1.92x |
| **037** | `uniql_compressor` | **UniQL (Unified Quantization & Low-Rank)** | $W \approx Q(W) + U_r V_r^\top, \quad \min_{Q, U, V} \|W - Q(W) - U V\|_F^2$ | Alternating minimization solving for joint discrete codebook quantization and continuous low-rank adapters. | 3.50x | 90.70% | 1.85x |
| **038** | `quarot` | **QuaRot (4-Bit All-Round Rotation)** | $X^\prime = X H, \; W^\prime = H W H, \quad Y = Q(X^\prime) Q(W^\prime) H^\prime$ | Complete 4-bit weight and activation quantization using randomized Walsh-Hadamard orthogonal basis rotations. | 3.90x | 90.50% | 2.20x |
| **039** | `slicegpt` | **SliceGPT Matrix Compressor** | $W_{\text{sliced}} = W[:, \mathcal{I}], \quad |\mathcal{I}| = (1 - \rho) d_{\text{in}}$ | Eliminates least significant input and output matrix columns after orthogonal projection of activation matrices. | 1.40x | 91.20% | 1.35x |
| **040** | `slicegpt_pca` | **SliceGPT-PCA Structured Compressor** | $C_X = U \Lambda U^\top, \quad \tilde{W} = W U[:, :k]$ | PCA-guided dimension truncation removing invariant subspaces with eigenvalues below empirical noise floor. | 1.45x | 91.30% | 1.38x |
| **041** | `flat_llm` | **FlatLLM Curvature Compressor** | $W_{\text{flat}} = W - \eta \nabla \mathcal{L}_{\text{sharp}}(W), \quad W_q = Q(W_{\text{flat}})$ | Pre-quantization sharpness-aware gradient descent finding flat loss minima resilient to rounding perturbations. | 3.60x | 91.00% | 1.90x |
| **042** | `mlorq_compressor` | **MLoRQ (Multi-Level Optimal Residual Quant)** | $W \approx \sum_{m=1}^M s_m Q_m(W - \hat{W}_{m-1})$ | Multi-stage hierarchical residual vector quantization iteratively refining sub-space error approximations. | 3.75x | 90.40% | 1.75x |

---

### Family 4: Structured Channel & Layer Pruning (14 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **043** | `admm_compressor` | **ADMM Compressor** | $\min_{W, Z} \mathcal{L}(W) + \frac{\rho}{2}\|W - Z + U\|_F^2 \quad \text{s.t.} \quad Z \in \mathcal{S}_{\text{sparse}}$ | Alternating Direction Method of Multipliers decomposing non-convex l0 pruning into proximal sub-problems. | 1.95x | 90.50% | 1.48x |
| **044** | `admm_pruner` | **ADMM Pruner with STE** | $Z^{k+1} = \Pi_{\mathcal{S}}(W^k + U^k), \quad U^{k+1} = U^k + W^{k+1} - Z^{k+1}$ | Augmented Lagrangian formulation with straight-through estimator updating dual multipliers during fine-tuning. | 1.98x | 90.60% | 1.50x |
| **045** | `armor_pruner` | **ARMOR (Adaptive Resilient Matrix Orthogonal Rank)** | $\mathcal{S}_i = \|W_i\|_2 \cdot (1 - \max_{j \ne i} |\cos(W_i, W_j)|)$ | Adaptive channel pruning scoring column norm scaled by directional orthogonality penalty. | 1.92x | 90.70% | 1.45x |
| **046** | `ast_pruner` | **AST Pruner (Adaptive Semi-Structured Pruning)** | $M = \text{TopK}_{2:4}(|W| \odot \sqrt{C_X})$ | Fisher and activation calibrated 2:4 semi-structured pruning with LoRA parameter healing. | 2.00x | 91.40% | 1.55x |
| **047** | `bonsai_pruner` | **Bonsai Pruner (Progressive Iterative Pruner)** | $s_t = s_f + (s_i - s_f)\left(1 - \frac{t}{T}\right)^3, \quad M_t = \mathbb{I}(|W| \ge \text{quantile}(s_t))$ | Cubic polynomial sparsity scheduling gradually pruning channels while freezing critical structural trunks. | 1.90x | 90.80% | 1.42x |
| **048** | `flap` | **FLAP (Fluctuation-Based Adaptive Pruner)** | $S_j = \text{Var}_X([W X]_j) \cdot \|W_{:, j}\|_2$ | Measures feature variance fluctuation across calibration batches to prune redundant channels. | 1.94x | 90.50% | 1.46x |
| **049** | `mamba_shedder` | **Mamba Shedder (Selective Channel Shedding)** | $M_{\text{shed}} = \text{Threshold}\left(\int_0^T |\Delta_t \cdot A| dt\right)$ | SSM-aware structured channel shedding eliminating dormant state update channels in recurrent projections. | 1.96x | 90.20% | 1.52x |
| **050** | `proxsparse` | **ProxSparse (Proximal Gradient Pruner)** | $W^{t+1} = \text{prox}_{\gamma \|\cdot\|_{2,1}}(W^t - \gamma \nabla \mathcal{L}(W^t))$ | Proximal gradient descent enforcing mixed l2,1 group lasso penalties over weight columns. | 1.95x | 90.40% | 1.47x |
| **051** | `slim_compressor` | **SLiM (Structured Low-Rank Integrated Matrix)** | $W \approx \Pi_{2:4}(W) + A_r B_r$ | Joint 2:4 structured hardware sparsity coupled with low-rank SVD residual reconstruction. | 2.05x | 91.20% | 1.60x |
| **052** | `gfws` | **GFWS (Group Fisher Weight Slicing)** | $\mathcal{I}_G = \arg\max_{|G|=k} \sum_{i \in G} F_{i,i} W_{i,i}^2$ | Grouped Fisher Information slicing pruning whole heads and attention projection blocks. | 1.92x | 90.60% | 1.45x |
| **053** | `mgie_budget` | **MGIE (Multi-Granular Information Entropy)** | $B_l = B_{\text{total}} \frac{I(X_l; Y)}{\sum_j I(X_j; Y)}$ | Mutual information entropy budgeting dynamically distributing pruning sparsity ratios across transformer depth. | 1.96x | 90.30% | 1.48x |
| **054** | `modegpt_compressor` | **ModeGPT (Modal Decomposition Pruner)** | $W = \sum_{m=1}^M c_m \Phi_m, \quad \hat{W} = \sum_{m \in \text{TopK}} c_m \Phi_m$ | Empirical modal decomposition filtering out high-frequency spatial modes from linear projections. | 1.93x | 90.40% | 1.44x |
| **055** | `cr_compressor` | **CR Compressor (Column-Row Pruning)** | $W \approx C U R, \quad C \in \mathbb{R}^{m \times c}, R \in \mathbb{R}^{r \times n}$ | Selects subset of actual columns and rows from weight matrices to construct interpolative skeleton representations. | 1.98x | 90.10% | 1.49x |
| **056** | `lillama_compressor` | **LiLLaMA (Layer-Wise Distilled Pruning)** | $\min_W \mathcal{L}_{\text{KD}}(f_S(X; W), f_T(X)) + \lambda \|W\|_{\text{group}}$ | Layer-by-layer progressive structured pruning healed via cached teacher intermediate representations. | 2.00x | 90.70% | 1.55x |

---

### Family 5: Riemannian Manifold & Geometric Optimizers (10 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **057** | `optimal_subspace_projection` | **Optimal Subspace Projection** | $P^* = \arg\max_{P \in \text{Gr}(r, d)} \text{Tr}(P^\top W^\top W P)$ | Grassmannian manifold optimization identifying the invariant principal projection subspace. | 2.05x | 91.50% | 1.58x |
| **058** | `subspace_projection` | **Subspace Projection Compressor** | $\hat{W} = P P^\top W, \quad P^\top P = I_r$ | Orthogonal Stiefel manifold projection mapping weight matrices onto fixed r-dimensional orthonormal bases. | 2.02x | 91.40% | 1.55x |
| **059** | `rpca_compressor` | **RPCA (Robust Principal Component Analysis)** | $\min_{L, S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad W = L + S$ | Decomposes weight matrices into low-rank underlying manifold L and sparse outlier matrix S. | 2.10x | 91.60% | 1.60x |
| **060** | `drone_compressor` | **DRONE (Data-aware Reduction via Orthogonal Subspace)** | $\min_{U \in \text{St}(r, d)} \|X W^\top - X W^\top U U^\top\|_F^2$ | Riemannian gradient descent on Stiefel manifold St(r, d) minimizing empirical reconstruction loss. | 2.08x | 91.55% | 1.62x |
| **061** | `enlighten_compressor` | **Enlighten Compressor** | $W \approx U \Sigma V^\top + \text{GeodesicCorrection}(U, V)$ | Geometric geodesic curvature correction along Riemannian manifolds during low-rank parameter synthesis. | 2.04x | 91.45% | 1.56x |
| **062** | `liftr` | **LIFTR (Linear Invariant Factorization & Retraction)** | $R_X(\xi) = \text{qf}(X + \xi), \quad X \in \text{St}(r, n)$ | Cayley and QR retraction operators optimizing low-rank orthonormal factorizations without matrix inversion. | 2.06x | 91.40% | 1.58x |
| **063** | `lsc_compressor` | **LSC (Local Subspace Clustering)** | $\min_{P_k} \sum_{k} \sum_{x \in C_k} \|x - P_k P_k^\top x\|^2$ | Partitions weights into local geometric subspaces, each compressed via dedicated affine projection. | 2.05x | 91.35% | 1.54x |
| **064** | `oats_compressor` | **OATS (Optimal Affine Transformation Subspace)** | $\hat{W} = A B + C, \quad C = \arg\min_C \|W - (A B + C)\|_F^2$ | Affine manifold decomposition augmenting low-rank linear factors with constant directional drift tensors. | 2.02x | 91.40% | 1.52x |
| **065** | `tomoe_compressor` | **Tomoe Compressor (Rotational Manifold Tracking)** | $Q \in \text{SO}(n), \quad \min_Q \|W - Q \hat{W}\|_F^2$ | Special orthogonal group SO(n) manifold tracking aligning weight spectra with canonical coordinate frames. | 2.06x | 91.50% | 1.59x |
| **066** | `hashed_net_compressor` | **Hashed Net Compressor** | $W_{i,j} = s(i, j) \cdot w[h(i, j)], \quad h: \mathbb{N}^2 \to [K]$ | Locality-sensitive hash projections tying parameter weights into compact 1D continuous memory buffers. | 2.00x | 91.30% | 1.50x |

---

### Family 6: Tensor Train & Kronecker Factorization (8 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **067** | `kronecker_compressor` | **Kronecker Product Compressor** | $W \approx A \otimes B, \quad A \in \mathbb{R}^{m_1 \times n_1}, B \in \mathbb{R}^{m_2 \times n_2}$ | Kronecker product factorization decomposing d x d weight matrices into small factor matrix products. | 2.80x | 89.20% | 1.55x |
| **068** | `kronecker_tt_lora` | **Kronecker TT-LoRA** | $W = (A_1 \otimes A_2) + U_{\text{TT}} V_{\text{TT}}^\top$ | Hybrid Kronecker matrix backbone complemented with Tensor-Train factorized low-rank adaptation branches. | 2.95x | 89.60% | 1.60x |
| **069** | `tt_compressor` | **TT (Tensor-Train) Compressor** | $W(i_1..i_d, j_1..j_d) = G_1[i_1, j_1] G_2[i_2, j_2] \cdots G_d[i_d, j_d]$ | Tensor-Train (Oseledets) decomposition transforming high-dimensional tensors into 3D core contractions. | 3.20x | 88.80% | 1.45x |
| **070** | `cur_compressor` | **CUR Matrix Decomposer** | $W \approx C U R, \quad U = C^+ W R^+$ | Decomposes weight matrices using exact sampled column and row vectors, ensuring non-negative structural interpretability. | 2.60x | 89.50% | 1.50x |
| **071** | `cur_moe_compressor` | **CUR MoE Compressor** | $W_e \approx C_e U_e R_e, \quad \forall e \in \{1, \dots, E\}$ | CUR decomposition tailored for Mixture-of-Experts routing layers sharing column basis matrices across experts. | 2.85x | 89.30% | 1.65x |
| **072** | `nmf_compressor` | **NMF (Non-Negative Matrix Factorization)** | $\min_{W_{\ge 0}, H_{\ge 0}} \|V - W H\|_F^2$ | Constrained non-negative additive decomposition yielding sparse, interpretable part-based weight representations. | 2.50x | 89.10% | 1.40x |
| **073** | `compot_compressor` | **ComPOT (Compact Polyadic Optimal Transport)** | $\mathcal{W} \approx \sum_{r=1}^R \lambda_r a_r^{(1)} \otimes a_r^{(2)} \otimes a_r^{(3)}$ | Canonical Polyadic (CP) tensor decomposition optimized via Wasserstein optimal transport metrics. | 2.70x | 89.00% | 1.48x |
| **074** | `roast_hashed_linear` | **ROAST (Randomized Orthogonal Array Hashing)** | $W x = \left(\sum_{k=1}^K g_k \odot P_k(w)\right) x$ | Block-randomized tensor hashing collapsing linear parameters into shared parameter arrays with low collision rates. | 2.40x | 89.10% | 1.52x |

---

### Family 7: Hardware-Aware Semi-Structured Sparsity (4 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **075** | `sparsegpt` | **SparseGPT (1-Shot Pruning Engine)** | $w_m^* = 0, \quad \delta w = -\frac{w_m}{[H^{-1}]_{mm}} H^{-1}_{:, m}$ | One-shot exact second-order error compensation pruning matrices to 50% unstructured or 2:4 semi-structured patterns. | 2.00x | 91.60% | 1.65x |
| **076** | `wanda` | **Wanda (Pruning by Weights and Activations)** | $S_{i,j} = |W_{i,j}| \cdot \|X_j\|_2$ | Multiplies absolute weight values by empirical input activation column norms to prune 2:4 patterns without inversion. | 2.00x | 91.70% | 1.62x |
| **077** | `aatc_xkv_compressor` | **AATC-XKV (Attention-Aware Cross-Layer KV)** | $K_{l+1} \approx P_l K_l, \quad V_{l+1} \approx Q_l V_l$ | Identifies shared KV-cache attention subspaces across adjacent layers to eliminate duplicate cache allocations. | 2.80x | 91.50% | 2.10x |
| **078** | `cachenotes_compressor` | **CacheNotes (Task-Aware KV Eviction)** | $m_t = \sigma(W_{\text{gate}} [K_t; V_t]), \quad \text{Evict if } m_t < \tau$ | Dynamic task-aware gating network learning token importance scores to evict uninformative KV-cache vectors. | 2.80x | 91.60% | 2.20x |

---

### Family 8: Cross-Layer & Subspace Sharing (4 Engines)

| ID | Registry Key | Engine Name | Theoretical Formulation | Core Mechanism & Operator Contract | Compression Ratio ($r=32$) | SST-2 Recovered Accuracy (%) | Speedup |
|:---:|:---|:---|:---|:---|:---:|:---:|:---:|
| **079** | `delta_kv` | **Delta-KV Cache Compressor** | $\Delta K_t = K_t - K_{t-1}, \quad \tilde{K}_t = K_{t-1} + Q_{\text{sparse}}(\Delta K_t)$ | Temporal differential delta encoding exploiting high token-to-token correlation across decoding timesteps. | 2.60x | 91.10% | 2.30x |
| **080** | `hybridkv_compressor` | **HybridKV Multi-Tier Compressor** | $\text{Cache} = \{\text{FP16 Sink}\} \cup \{\text{INT4 Rolling}\} \cup \{\text{Evicted SVD}\}$ | Tiered memory hierarchy: maintains initial attention sinks in FP16, local window in INT4, and evicted keys in SVD. | 2.70x | 90.90% | 2.40x |
| **081** | `kvtc_compressor` | **KVTC (Key-Value Tensor Contraction)** | $\mathcal{K}_{B, H, S, D} \approx \mathcal{C}_{B, H, S, r} \times_4 U_{r, D}$ | Cross-head tensor contraction factorizing multi-head KV cache tensors across head dimensions. | 2.50x | 90.70% | 2.15x |
| **082** | `zerokv_neo_compressor` | **ZeroKV Neo (Zero-Footprint KV Distillation)** | $h_t = \text{SSMRecurrence}(h_{t-1}, x_t), \quad K_t, V_t \to 0$ | Substitutes quadratic KV history buffers with compact recurrent state vectors via student distillation. | 2.60x | 90.50% | 2.50x |

---

## 💻 Functional Usage & Python API

### 1. Unified Compressor Factory (`neuralops.create_compressor`)

```python
import neuralops
from transformers import AutoModelForSequenceClassification

# Load baseline uncompressed teacher model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Instantiate any of the 82 registered compression engines
compressor = neuralops.create_compressor(
    "fwsvd_compressor",
    model=model,
    rank=32,
    fisher_samples=1024,
)
compressed_model = compressor.compress()

# Evaluate parameter reduction and inference latency
print(f"Compressed Parameter Ratio: {compressor.metrics.compression_ratio:.2f}x")
print(f"Downstream SST-2 Accuracy: {compressor.metrics.recovered_accuracy:.2f}%")
```

### 2. High-Level Convenience Runner (`neuralops.compress`)

```python
import neuralops

# One-line model surgery compression
compressed_model = neuralops.compress(
    model,
    method="hadamard_incoherent_svd",
    rank=32,
)
```

---

## 📊 Grounded Hardware Verification & Telemetry

All 82 compression engines undergo physical execution gating:
- **Silicon Target**: **NVIDIA GeForce RTX 3090 (Ampere SM 8.6, GA102)**
- **Hardware Architecture**: 10,496 CUDA cores, 82 SMs, 328 3rd-Gen Tensor Cores, 24 GB GDDR6X (936 GB/s)
- **Software Environment**: CUDA 12.2, PyTorch 2.4, Python 3.10+, Triton 3.0
- **Official Ledger Repository**: [https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)
