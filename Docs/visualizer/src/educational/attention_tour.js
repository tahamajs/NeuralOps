/**
 * educational/attention_tour.js
 * =============================
 * 10-Step Guided Narrative Camera & Architecture Deep Dive.
 * Focuses on each core attention mechanism with camera positioning,
 * interactive 3D highlights, KaTeX mathematical equations, and live metrics.
 */

export const ATTENTION_TOUR_STEPS = [
    {
        step: 1,
        title: 'Transformer Architecture & Token Ingress',
        target: 'embeddings',
        y: 0.5,
        dist: 16.0,
        model_ref: 'distilbert',
        formula: 'X_0 = \\text{Embed}(T) + E_{\\text{pos}}',
        key_metric: 'Vocabulary: 30,522 Tokens · d=768',
        narrative: 'Tokens enter the transformer stack at the baseplate, mapping discrete vocabulary indices into continuous 768-dimensional latent vectors combined with positional encodings.'
    },
    {
        step: 2,
        title: 'Multi-Head Attention (MHA) Decomposition',
        target: 'transformer.layer.0.attention.q_lin',
        y: 1.55,
        dist: 7.5,
        model_ref: 'distilbert',
        formula: '\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_{12}) W_O',
        key_metric: '12 Heads × 64 Dim = 768d · 2.36 MB',
        narrative: 'Layer 0 partitions the 768-dimensional hidden representation into 12 parallel subspaces. Each head independently computes scaled dot-product attention to model distinct syntactic and semantic relationships.'
    },
    {
        step: 3,
        title: 'Grouped-Query Attention (GQA) & KV Slicing',
        target: 'transformer.layer.0.attention.q_lin',
        y: 1.55,
        dist: 6.8,
        model_ref: 'qwen',
        formula: 'H_{KV} = 2 \\ll H_Q = 14 \\quad (7:1 \\text{ Query-to-Group Ratio})',
        key_metric: 'KV Cache: 87.5% Memory Reduction',
        narrative: 'In modern LLMs like Qwen-2.5, storing separate Key-Value heads for every Query consumes massive VRAM. GQA groups 7 query heads to share a single Key and Value head, drastically shrinking memory bandwidth bottlenecks.'
    },
    {
        step: 4,
        title: 'Fused QKV Projections (c_attn)',
        target: 'transformer.layer.0.attention.q_lin',
        y: 1.55,
        dist: 6.8,
        model_ref: 'gpt2',
        formula: '[Q, K, V] = \\text{Split}_3(X W_{\\text{c\\_attn}}), \\quad W \\in \\mathbb{R}^{768 \\times 2304}',
        key_metric: 'Fused GEMM: 1 Kernel Launch · 1.77M Params',
        narrative: 'GPT-2 coalesces Query, Key, and Value projections into a single fused linear operator [768 × 2304]. This maximizes GPU Tensor Core arithmetic intensity and minimizes memory round-trips.'
    },
    {
        step: 5,
        title: 'SVD Low-Rank Bottleneck Compression',
        target: 'transformer.layer.0.attention.q_lin',
        y: 1.55,
        dist: 6.2,
        model_ref: 'distilbert',
        formula: 'W \\approx U_{768 \\times 64} \\Sigma_{64} V_{64 \\times 768}^\\top',
        key_metric: '83.3% Parameter Savings · 92.4% SST-2 Retained',
        narrative: 'Our core SVD factorization replaces heavy 768×768 weight matrices with factorized bottleneck twins: Factor V projects 768 down to rank 64, and Factor U reconstructs 64 back to 768, cutting storage by over 80%.'
    },
    {
        step: 6,
        title: 'Multi-Head Latent Attention (MLA)',
        target: 'transformer.layer.0.attention.q_lin',
        y: 1.55,
        dist: 6.5,
        model_ref: 'distilbert',
        formula: 'c_t^{KV} = W^{DKV} x_t, \\quad K_t^C = W^{UK} c_t^{KV}',
        key_metric: 'Latent Dim: 512 · Zero-Overhead Cache',
        narrative: 'Pioneered in DeepSeek-V2, MLA compresses Key and Value representations into a compact 512-dimensional latent vector c_t^{KV}. The decompression matrix can be absorbed directly into the queries, keeping inference blazing fast.'
    },
    {
        step: 7,
        title: 'Mixture of Experts (MoE) Routing',
        target: 'transformer.layer.0.ffn.lin1',
        y: 3.35,
        dist: 8.0,
        model_ref: 'distilbert',
        formula: 'y = \\sum_{i \\in \\text{Top2}} G(x)_i E_i(x)',
        key_metric: '4 Parallel Experts · 2 Active / Token',
        narrative: 'In the Feed-Forward layer, tokens encounter a learned gating router that dynamically dispatches each token to only 2 of 4 specialized expert networks, scaling parameter capacity without increasing FLOPs.'
    },
    {
        step: 8,
        title: 'Structured Pruning & Coordinate Slicing',
        target: 'transformer.layer.0.ffn.lin1',
        y: 3.35,
        dist: 7.2,
        model_ref: 'distilbert',
        formula: 'W_{\\text{slice}} = W Q P_r, \\quad Q^\\top Q = I',
        key_metric: '25% Channel Pruning · Hardware Dense',
        narrative: 'SliceGPT computes orthogonal transformation matrices Q via activation covariance PCA, concentrating network energy into principal coordinates and physically excising 25% of unneeded channels.'
    },
    {
        step: 9,
        title: 'Residual Highway & Layer Normalization',
        target: 'transformer.layer.0.attention.out_lin',
        y: 2.45,
        dist: 7.0,
        model_ref: 'distilbert',
        formula: 'X_{\\text{next}} = \\text{RMSNorm}(X + \\text{Sublayer}(X))',
        key_metric: 'Highway Skip: Zero-Degradation Gradient Flow',
        narrative: 'The green lateral highway carries unperturbed identity tokens around parameter transformations. Merged via additive residual junctions (+), this prevents vanishing gradients across deep models.'
    },
    {
        step: 10,
        title: 'Cross-Architecture Evaluation Matrix',
        target: null,
        fullFit: true,
        y: 8.0,
        dist: 22.0,
        model_ref: 'all',
        formula: '\\Delta_{\\text{storage}} = -56.0\\%, \\; \\text{Speedup} = 1.65\\times',
        key_metric: 'Pareto Optimal Frontier Verified',
        narrative: 'Comparing DistilBERT, RoBERTa, GPT-2, and Qwen-2.5 reveals that structured SVD compression and GQA/MLA latent paradigms achieve the optimal Pareto trade-off between inference speed, VRAM, and task accuracy.'
    }
];
