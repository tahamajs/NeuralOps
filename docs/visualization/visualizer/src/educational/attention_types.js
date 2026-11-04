/**
 * educational/attention_types.js
 * ==============================
 * Comprehensive specifications for 13 Attention Mechanisms:
 * - Architectural parameters, tensor dimensions, computational graphs
 * - Theoretical KV Cache sizing and autoregressive decoding memory footprints
 * - Formulated mathematical expressions (KaTeX ready)
 * - Automatic detection heuristic from module metadata & active model
 */

export const ATTENTION_VARIANTS = {
    mha: {
        id: 'mha',
        name: 'Multi-Head Attention (MHA)',
        family: 'standard',
        icon: '🔱',
        color: '#3B82F6',
        causal: false,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: '\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W_O',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Vaswani et al. (2017). Splits hidden dimension into H independent subspaces, allowing the model to jointly attend to information from different representation subspaces.',
        computation_graph: [
            { step: 1, title: 'Linear Projections', op: 'Q = X W_Q, \\; K = X W_K, \\; V = X W_V', desc: 'Project input tensor X into Query, Key, and Value subspaces.' },
            { step: 2, title: 'Head Splitting', op: 'Q_h = \\text{Split}(Q, H), \\; K_h = \\text{Split}(K, H)', desc: 'Partition hidden dimension 768 into 12 heads of dimension 64.' },
            { step: 3, title: 'Scaled Dot-Product', op: 'A_h = \\text{Softmax}\\left(\\frac{Q_h K_h^\\top}{\\sqrt{d_k}}\\right)', desc: 'Compute pairwise token affinity scores scaled by 1/√64 = 0.125.' },
            { step: 4, title: 'Value Aggregation', op: 'O_h = A_h V_h', desc: 'Weighted summation over Value vectors based on attention probability weights.' },
            { step: 5, title: 'Output Projection', op: '\\text{Out} = \\text{Concat}(O_1, \\dots, O_H) W_O', desc: 'Concatenate head outputs and project back to hidden dimension 768.' }
        ],
        sample_attention_matrix: [
            [0.72, 0.14, 0.09, 0.05],
            [0.18, 0.62, 0.15, 0.05],
            [0.08, 0.19, 0.63, 0.10],
            [0.04, 0.09, 0.24, 0.63]
        ]
    },

    mqa: {
        id: 'mqa',
        name: 'Multi-Query Attention (MQA)',
        family: 'kv_compressed',
        icon: '⚡',
        color: '#F59E0B',
        causal: true,
        q_heads: 12,
        kv_heads: 1,
        d_head: 64,
        formula: '\\text{MQA}(Q, K_1, V_1) = \\text{Concat}(\\text{Softmax}(Q_h K_1^\\top / \\sqrt{d_k}) V_1) W_O',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 64]',
            W_V: '[768 × 64]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 1 * 64 * 2 * batch,
        summary: 'Shazeer (2019). Uses multiple query heads but shares a single Key and Value head across all queries, drastically reducing KV cache size by 12x with minor quality degradation.',
        computation_graph: [
            { step: 1, title: 'Asymmetric Projections', op: 'Q \\in \\mathbb{R}^{B \\times S \\times H \\times d}, \\; K,V \\in \\mathbb{R}^{B \\times S \\times 1 \\times d}', desc: 'Project queries into H heads, but project Keys and Values into exactly 1 shared head.' },
            { step: 2, title: 'Broadcast Attention', op: 'A_h = \\text{Softmax}\\left(\\frac{Q_h K^\\top}{\\sqrt{d_k}}\\right)', desc: 'Broadcast the single Key head across all H Query heads.' },
            { step: 3, title: 'Shared Value Mixing', op: 'O_h = A_h V', desc: 'Mix shared Value vectors using per-head attention matrices.' },
            { step: 4, title: 'Projection Out', op: '\\text{Out} = \\text{Concat}(O_1, \\dots, O_H) W_O', desc: 'Linear projection back to model dimension.' }
        ],
        sample_attention_matrix: [
            [0.85, 0.00, 0.00, 0.00],
            [0.35, 0.65, 0.00, 0.00],
            [0.20, 0.30, 0.50, 0.00],
            [0.15, 0.15, 0.20, 0.50]
        ]
    },

    gqa: {
        id: 'gqa',
        name: 'Grouped-Query Attention (GQA)',
        family: 'kv_compressed',
        icon: '🎯',
        color: '#2DD4BF',
        causal: true,
        q_heads: 14,
        kv_heads: 2,
        d_head: 64,
        formula: '\\text{GQA}(Q_g, K_g, V_g) = \\text{Concat}(\\text{Attn}(Q_{g,h}, K_g, V_g)) W_O',
        projections: {
            W_Q: '[896 × 896]',
            W_K: '[896 × 128]',
            W_V: '[896 × 128]',
            W_O: '[896 × 896]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 14) => 2 * layers * 2 * 64 * 2 * batch,
        summary: 'Ainslie et al. (2023, Qwen / LLaMA-3). Divides query heads into G groups (e.g. 7 queries per group), sharing 1 Key and 1 Value head per group to achieve MQA speed with MHA accuracy.',
        computation_graph: [
            { step: 1, title: 'Grouped Projections', op: 'Q \\in \\mathbb{R}^{H_Q \\times d}, \\; K,V \\in \\mathbb{R}^{H_{KV} \\times d}, \\; H_{KV} = H_Q / G', desc: 'Project into 14 Query heads and 2 Key/Value head groups.' },
            { step: 2, title: 'Group Key Replication', op: 'K_{\\text{rep}} = \\text{Repeat}(K, G)', desc: 'Virtually repeat each Key/Value head across its 7 assigned Query heads.' },
            { step: 3, title: 'Grouped Attention', op: 'A_{g,h} = \\text{Softmax}\\left(\\frac{Q_{g,h} K_g^\\top}{\\sqrt{d_k}} + M\\right)', desc: 'Compute causal scaled dot-product attention per group.' },
            { step: 4, title: 'Aggregation & Output', op: '\\text{Out} = \\text{Concat}(O_{1..H_Q}) W_O', desc: 'Concatenate all 14 head outputs and project through W_O.' }
        ],
        sample_attention_matrix: [
            [1.00, 0.00, 0.00, 0.00],
            [0.25, 0.75, 0.00, 0.00],
            [0.10, 0.30, 0.60, 0.00],
            [0.05, 0.15, 0.25, 0.55]
        ]
    },

    mla: {
        id: 'mla',
        name: 'Multi-Head Latent Attention (MLA)',
        family: 'latent_compressed',
        icon: '💎',
        color: '#A855F7',
        causal: true,
        q_heads: 16,
        kv_heads: 1,
        d_head: 64,
        latent_dim: 512,
        formula: 'c_t^{KV} = W^{DKV} x_t, \\quad K_t^C = W^{UK} c_t^{KV}, \\; V_t^C = W^{UV} c_t^{KV}',
        projections: {
            W_DKV: '[768 × 512] (Down-projection)',
            W_UK: '[512 × 1024] (Key Decompress)',
            W_UV: '[512 × 1024] (Value Decompress)',
            W_O: '[1024 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 16) => layers * 512 * 2 * batch,
        summary: 'DeepSeek-V2 / DeepSeek-V3. Compresses Key and Value representations into a compact low-dimensional latent vector c_t^{KV}, drastically shrinking KV cache during decoding.',
        computation_graph: [
            { step: 1, title: 'KV Compression', op: 'c_t^{KV} = W^{DKV} x_t', desc: 'Project hidden states into compact 512-d latent subspace stored in cache.' },
            { step: 2, title: 'RoPE Decoupling', op: 'q_t^R = \\text{RoPE}(W^{QR} x_t), \\; k_t^R = \\text{RoPE}(W^{KR} x_t)', desc: 'Compute separate positional queries and keys to avoid decompressing cache for RoPE.' },
            { step: 3, title: 'Matrix Absorption', op: 'q_t^{C\\prime} = q_t^C W^{UK}, \\; A = \\text{Softmax}(q_t^{C\\prime} c_t^{KV} + q_t^R k_t^R)', desc: 'Absorb decompression matrix W^{UK} into query matrix multiplication.' },
            { step: 4, title: 'Value Synthesis', op: 'O = A (c^{KV} W^{UV}) W_O', desc: 'Accumulate weighted latent values and project to output.' }
        ],
        sample_attention_matrix: [
            [0.90, 0.00, 0.00, 0.00],
            [0.40, 0.60, 0.00, 0.00],
            [0.15, 0.25, 0.60, 0.00],
            [0.08, 0.12, 0.30, 0.50]
        ]
    },

    fused_qkv: {
        id: 'fused_qkv',
        name: 'Fused QKV Attention (GPT-2 c_attn)',
        family: 'fused_kernel',
        icon: '📦',
        color: '#10B981',
        causal: true,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: '[Q, K, V] = \\text{Split}_{3}(X W_{\\text{c\\_attn}}), \\quad W_{\\text{c\\_attn}} \\in \\mathbb{R}^{d \\times 3d}',
        projections: {
            W_c_attn: '[768 × 2304] (Fused)',
            W_c_proj: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Radford et al. (2019, GPT-2). Computes Query, Key, and Value projections in a single monolithic gemm operation to maximize GPU kernel occupancy and minimize memory latency.',
        computation_graph: [
            { step: 1, title: 'Monolithic GEMM', op: 'Z = X W_{\\text{c\\_attn}} + b_{\\text{c\\_attn}}', desc: 'Compute [768 × 2304] matrix product in a single high-efficiency CUDA kernel.' },
            { step: 2, title: 'Tensor Slicing', op: 'Q, K, V = \\text{Chunk}(Z, 3, \\text{dim}=-1)', desc: 'Slice output into Query, Key, and Value tensors.' },
            { step: 3, title: 'Causal Masking', op: 'S = \\frac{Q K^\\top}{\\sqrt{64}} + M_{\\text{causal}}', desc: 'Apply upper-triangular -inf mask to enforce autoregressive property.' },
            { step: 4, title: 'Projection Out', op: '\\text{Out} = (\\text{Softmax}(S) V) W_{\\text{c\\_proj}}', desc: 'Project contextual value mix back to 768.' }
        ],
        sample_attention_matrix: [
            [1.00, 0.00, 0.00, 0.00],
            [0.32, 0.68, 0.00, 0.00],
            [0.12, 0.28, 0.60, 0.00],
            [0.05, 0.15, 0.30, 0.50]
        ]
    },

    sliding_window: {
        id: 'sliding_window',
        name: 'Sliding Window Attention (SWA)',
        family: 'sparse_local',
        icon: '🪟',
        color: '#06B6D4',
        causal: true,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        window_size: 512,
        formula: 'A_{i,j} = \\begin{cases} \\text{Softmax}\\left(\\frac{q_i k_j^\\top}{\\sqrt{d}}\\right) & \\text{if } 0 \\le i - j < W \\\\ 0 & \\text{otherwise} \\end{cases}',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * Math.min(seqLen, 512) * batch,
        summary: 'Beltagy et al. (Longformer) / Mistral. Restricts attention to a local context window W around each token, reducing memory complexity from O(N^2) to O(N * W).',
        computation_graph: [
            { step: 1, title: 'Local Window Bounds', op: 'j \\in [\\max(0, i - W), i]', desc: 'Limit key lookups to the preceding W tokens.' },
            { step: 2, title: 'Banded Dot Product', op: 'S_{i, j} = q_i k_j^\\top / \\sqrt{d_k}', desc: 'Compute dot-products only within the local band diagonal.' },
            { step: 3, title: 'Multi-layer Receptive Field', op: '\\text{RF} = L \\times W', desc: 'Information cascades across L layers, yielding an effective receptive field of L * W.' }
        ],
        sample_attention_matrix: [
            [1.00, 0.00, 0.00, 0.00],
            [0.45, 0.55, 0.00, 0.00],
            [0.00, 0.40, 0.60, 0.00],
            [0.00, 0.00, 0.35, 0.65]
        ]
    },

    sparse_attention: {
        id: 'sparse_attention',
        name: 'Sparse / Block-Sparse Attention',
        family: 'sparse_local',
        icon: '🕸️',
        color: '#8B5CF6',
        causal: true,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: 'A_{i,j} = \\text{Softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d}} \\odot M_{\\text{sparse}}\\right)',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => Math.round(2 * layers * 12 * 64 * 2 * seqLen * 0.25 * batch),
        summary: 'Child et al. (OpenAI Sparse Transformer) / BigBird. Uses a predefined sparsity pattern (local window + strided global tokens) to achieve sub-quadratic complexity.',
        computation_graph: [
            { step: 1, title: 'Sparsity Mask Topology', op: 'M_{i,j} \\in \\{0, -\\infty\\}', desc: 'Combine local diagonal blocks with strided interval columns.' },
            { step: 2, title: 'Block Gather GEMM', op: 'S_{\\text{block}} = Q_{\\text{blk}} K_{\\text{blk}}^\\top', desc: 'Execute sparse block-matrix multiplication.' }
        ],
        sample_attention_matrix: [
            [0.80, 0.00, 0.20, 0.00],
            [0.10, 0.70, 0.00, 0.20],
            [0.20, 0.00, 0.65, 0.15],
            [0.05, 0.15, 0.00, 0.80]
        ]
    },

    linear_attention: {
        id: 'linear_attention',
        name: 'Linear Kernel Attention',
        family: 'linear_complexity',
        icon: '📈',
        color: '#EC4899',
        causal: false,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: '\\text{Attn}(Q, K, V) = \\frac{\\phi(Q) (\\phi(K)^\\top V)}{\\phi(Q) \\sum_j \\phi(K_j)^\\top}',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => layers * 64 * 64 * 4 * batch,
        summary: 'Katharopoulos et al. (2020) / Performer. Replaces softmax with kernel feature maps phi(x), enabling associative matrix multiplication that computes in O(N) linear time.',
        computation_graph: [
            { step: 1, title: 'Feature Mapping', op: '\\phi(x) = \\text{ELU}(x) + 1', desc: 'Apply non-negative kernel feature mapping to Queries and Keys.' },
            { step: 2, title: 'Associative Order Swap', op: 'S = \\phi(K)^\\top V \\in \\mathbb{R}^{d \\times d}', desc: 'Compute Key-Value context matrix first: O(N * d^2) instead of O(N^2 * d).' },
            { step: 3, title: 'Query Multiplication', op: 'O = \\phi(Q) S', desc: 'Multiply queries by precomputed context matrix.' }
        ],
        sample_attention_matrix: [
            [0.35, 0.25, 0.22, 0.18],
            [0.26, 0.32, 0.24, 0.18],
            [0.20, 0.24, 0.34, 0.22],
            [0.18, 0.20, 0.26, 0.36]
        ]
    },

    flash_attention: {
        id: 'flash_attention',
        name: 'FlashAttention (Tiled IO-Aware)',
        family: 'hardware_optimized',
        icon: '⚡',
        color: '#EAB308',
        causal: true,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: 'O = \\text{TiledSoftmaxGEMM}(Q, K, V) \\quad [\\text{Exact Softmax in GPU SRAM}]',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Dao et al. (2022, FlashAttention-1/2/3). Restructures exact attention computation into SRAM tiles using online softmax normalization, eliminating intermediate HBM reads/writes.',
        computation_graph: [
            { step: 1, title: 'SRAM Block Tiling', op: 'B_r = \\lceil M / 4d \\rceil, \\; B_c = \\lceil M / 4d \\rceil', desc: 'Partition Q into blocks of size B_r and K, V into blocks of size B_c.' },
            { step: 2, title: 'Online Softmax Accumulation', op: 'm_{\\text{new}} = \\max(m_{\\text{prev}}, \\text{rowmax}(S_{ij}))', desc: 'Track running max and sum statistics without materializing the full N x N matrix.' },
            { step: 3, title: 'Direct Value Accumulation', op: 'O_i \\leftarrow O_i \\cdot \\alpha + P_{ij} V_j', desc: 'Rescale running output vector directly in fast on-chip SRAM registers.' }
        ],
        sample_attention_matrix: [
            [1.00, 0.00, 0.00, 0.00],
            [0.28, 0.72, 0.00, 0.00],
            [0.10, 0.25, 0.65, 0.00],
            [0.06, 0.14, 0.28, 0.52]
        ]
    },

    cross_attention: {
        id: 'cross_attention',
        name: 'Cross-Attention (Encoder-Decoder)',
        family: 'multimodal_encdec',
        icon: '🔀',
        color: '#6366F1',
        causal: false,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: '\\text{CrossAttn}(Q_{\\text{dec}}, K_{\\text{enc}}, V_{\\text{enc}}) = \\text{Softmax}\\left(\\frac{Q_{\\text{dec}} K_{\\text{enc}}^\\top}{\\sqrt{d_k}}\\right) V_{\\text{enc}}',
        projections: {
            W_Q: '[768 × 768] (Decoder States)',
            W_K: '[768 × 768] (Encoder States)',
            W_V: '[768 × 768] (Encoder States)',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Vaswani et al. (T5 / Whisper / Stable Diffusion). Queries come from the target decoder sequence, while Keys and Values come from the conditioning encoder representations.',
        computation_graph: [
            { step: 1, title: 'Heterogeneous Inputs', op: 'Q = X_{\\text{dec}} W_Q, \\; K = Z_{\\text{enc}} W_K, \\; V = Z_{\\text{enc}} W_V', desc: 'Project decoder queries against external encoder representations.' },
            { step: 2, title: 'Bipartite Token Matching', op: 'A_{i,j} = \\text{Softmax}(Q_i K_j^\\top / \\sqrt{d_k})', desc: 'Compute alignment weights between decoder token i and encoder token j.' }
        ],
        sample_attention_matrix: [
            [0.45, 0.35, 0.15, 0.05],
            [0.12, 0.55, 0.25, 0.08],
            [0.08, 0.18, 0.62, 0.12],
            [0.05, 0.10, 0.35, 0.50]
        ]
    },

    self_attention: {
        id: 'self_attention',
        name: 'Bidirectional Self-Attention (BERT / RoBERTa)',
        family: 'standard',
        icon: '↔️',
        color: '#3B82F6',
        causal: false,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: '\\text{SelfAttn}(X) = \\text{Softmax}\\left(\\frac{(X W_Q)(X W_K)^\\top}{\\sqrt{d_k}}\\right) (X W_V) W_O',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 0, // No KV cache in encoder models
        summary: 'Devlin et al. (BERT) / Liu et al. (RoBERTa). All tokens attend unconstrained to all past and future positions in the sequence, optimal for masked language modeling and classification.',
        computation_graph: [
            { step: 1, title: 'Full Bidirectional Matrix', op: 'A_{i,j} > 0 \\quad \\forall i, j \\in [1, N]', desc: 'Every token attends to every token without causal masking.' },
            { step: 2, title: 'Dense Context Mixing', op: 'O_i = \\sum_{j=1}^N A_{ij} V_j', desc: 'Pool representations across the entire sentence.' }
        ],
        sample_attention_matrix: [
            [0.55, 0.20, 0.15, 0.10],
            [0.22, 0.50, 0.18, 0.10],
            [0.12, 0.22, 0.52, 0.14],
            [0.08, 0.12, 0.25, 0.55]
        ]
    },

    causal_attention: {
        id: 'causal_attention',
        name: 'Autoregressive Causal Attention',
        family: 'autoregressive',
        icon: '➡️',
        color: '#14B8A6',
        causal: true,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        formula: 'A_{i,j} = \\text{Softmax}\\left(\\frac{q_i k_j^\\top}{\\sqrt{d}} + M_{i,j}\\right), \\quad M_{i,j} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}',
        projections: {
            W_Q: '[768 × 768]',
            W_K: '[768 × 768]',
            W_V: '[768 × 768]',
            W_O: '[768 × 768]'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Strict lower-triangular attention mask ensuring token at position i cannot access information from tokens at positions j > i, preserving autoregressive generation invariants.',
        computation_graph: [
            { step: 1, title: 'Mask Application', op: 'S = (Q K^\\top / \\sqrt{d}) + M_{\\text{causal}}', desc: 'Add -inf to upper triangle.' },
            { step: 2, title: 'Probability Normalization', op: 'A = \\text{Softmax}(S)', desc: 'Upper triangle entries strictly exponentiate to 0.' }
        ],
        sample_attention_matrix: [
            [1.00, 0.00, 0.00, 0.00],
            [0.40, 0.60, 0.00, 0.00],
            [0.20, 0.35, 0.45, 0.00],
            [0.10, 0.20, 0.30, 0.40]
        ]
    },

    svd_linear_attention: {
        id: 'svd_linear_attention',
        name: 'SVD Low-Rank Attention (SOA / SVD)',
        family: 'low_rank_compressed',
        icon: '📉',
        color: '#2DD4BF',
        causal: false,
        q_heads: 12,
        kv_heads: 12,
        d_head: 64,
        rank: 64,
        formula: 'W \\approx U_r \\Sigma_r V_r^\\top = A B^\\top, \\quad r \\ll d_{\\text{model}}',
        projections: {
            W_V_factor: '[768 × 64] (Down-projection)',
            W_U_factor: '[64 × 768] (Up-projection)',
            Compression: '83.3% parameter reduction'
        },
        kv_cache_per_token_bytes: (seqLen, batch = 1, layers = 12) => 2 * layers * 12 * 64 * 2 * batch,
        summary: 'Approximates full projection matrices via Singular Value Decomposition bottleneck factorization W = U * V, reducing parameter counts and memory by over 80% while retaining >90% accuracy.',
        computation_graph: [
            { step: 1, title: 'Eckart-Young Truncation', op: 'W_{m \\times n} \\approx U_{m \\times r} \\Sigma_{r \\times r} V_{r \\times n}^\\top', desc: 'Optimal rank-r approximation under Frobenius and spectral norms.' },
            { step: 2, title: 'Dual-Layer Bottleneck', op: 'y = (x B^\\top) A^\\top = x (V \\Sigma^{1/2}) (\\Sigma^{1/2} U^\\top)', desc: 'Replaces single dense linear layer with 2 sequential low-rank matrix multiplications.' },
            { step: 3, title: 'Inference Acceleration', op: '\\text{FLOPs} = 2 \\times (m \\times r + r \\times n) \\ll 2mn', desc: 'Drastically reduces memory bandwidth overhead.' }
        ],
        sample_attention_matrix: [
            [0.68, 0.16, 0.10, 0.06],
            [0.19, 0.58, 0.17, 0.06],
            [0.10, 0.20, 0.58, 0.12],
            [0.05, 0.10, 0.25, 0.60]
        ]
    }
};

/**
 * Detects which attention variant corresponds to the given module & model.
 * @param {object} moduleMeta - Module metadata object
 * @param {string} currentModel - Active model identifier
 * @returns {object} Attention variant definition
 */
export function detectAttentionVariant(moduleMeta = {}, currentModel = 'distilbert') {
    const name = (moduleMeta.name || moduleMeta.id || '').toLowerCase();
    const type = (moduleMeta.type || '').toLowerCase();
    const paradigm = (moduleMeta.paradigm || '').toLowerCase();

    if (name.includes('mla') || type.includes('mla') || paradigm.includes('mla')) {
        return ATTENTION_VARIANTS.mla;
    }
    if (name.includes('gqa') || type.includes('gqa') || currentModel === 'qwen') {
        return ATTENTION_VARIANTS.gqa;
    }
    if (name.includes('mqa') || type.includes('mqa')) {
        return ATTENTION_VARIANTS.mqa;
    }
    if (name.includes('c_attn') || currentModel === 'gpt2') {
        return ATTENTION_VARIANTS.fused_qkv;
    }
    if (name.includes('sliding') || name.includes('window') || name.includes('swa')) {
        return ATTENTION_VARIANTS.sliding_window;
    }
    if (name.includes('sparse')) {
        return ATTENTION_VARIANTS.sparse_attention;
    }
    if (name.includes('svd') || paradigm === 'svd' || name.includes('soa')) {
        return ATTENTION_VARIANTS.svd_linear_attention;
    }
    if (name.includes('flash')) {
        return ATTENTION_VARIANTS.flash_attention;
    }
    if (name.includes('linear')) {
        return ATTENTION_VARIANTS.linear_attention;
    }
    if (name.includes('cross')) {
        return ATTENTION_VARIANTS.cross_attention;
    }
    if (currentModel === 'roberta') {
        return ATTENTION_VARIANTS.self_attention;
    }
    return ATTENTION_VARIANTS.mha;
}
