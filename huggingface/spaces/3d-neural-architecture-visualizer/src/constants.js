/**
 * constants.js
 * ============
 * Central configuration and constant definitions for the 3D Neural Architecture Visualizer.
 * All magic numbers, color palettes, model specifications, and animation profiles live here.
 * Pure ES module with zero external dependencies (Node & Browser compatible).
 */

/**
 * Global visual tuning parameters and layout thresholds.
 */
export const VISUAL_CONFIG = {
    // ── Token Physics & Particle Flow ─────────────────────────
    TOKENS: {
        count: 360,             // Total instanced particles per model column
        baseSpeed: 0.026,       // Vertical velocity (units per frame at 1.0x speed)
        speedVariance: 0.034,   // Stochastic particle speed jitter
        trailMultiplier: 1.8,   // Trail distance offset multiplier
        trailScale: 0.65,       // Trail particle relative radius
        lateralNoiseRange: 1.4, // Max lateral stochastic deviation
        layerStride: 3.2,       // Vertical Y-distance between layer centers
        sphereRadius: 0.1,      // Main token sphere radius
        trailRadius: 0.065,     // Trail sphere radius
        sphereSegments: 8,      // Sphere mesh resolution
        residualRatio: 0.25,    // 25% of tokens bypass through residual highways
    },

    // ── Dual-Viewport Layout Geometry ─────────────────────────
    LAYOUT: {
        baselineX: -4.5,        // Uncompressed Baseline column X offset
        compressedX: 4.5,       // Compressed column X offset
        curveBulge: 0.45,       // Bypass conduit lateral bulge
        bypassOffset: 1.85,     // Residual skip conduit lateral spacing
        maxGraphHeight: 16.0,   // Initial scene height threshold
    },

    // ── Semantic Zoom & Level-of-Detail (LOD) ─────────────────
    ZOOM: {
        macroThreshold: 34,     // camDist > 34: Layers collapse to macro bounding blocks
        transitionStart: 28,    // 24 < camDist <= 34: Granular nodes fade in
        microThreshold: 24,     // camDist <= 24: Full matrix tensor resolution & badges
        badgeFadeRange: 5,      // Delta distance units over which badges reach full opacity
    },

    // ── Animation & Transition Timings ────────────────────────
    ANIM: {
        pulseDuration: 320,     // Node activation flash duration in ms
        cameraLerp: 0.07,       // OrbitControls damping factor
        tourStepDuration: 3200, // Duration per auto-tour stage in ms
    },

    // ── Color Palette (Hex integer codes for Three.js PBR Materials) ───
    COLORS: {
        sceneBackground: 0x060a12,
        hemiSky: 0xdbeafe,
        hemiGround: 0x0f172a,
        mainLight: 0xfff7ed,
        cyanRim: 0x53ecd9,
        purpleRim: 0xc895ea,
        
        // Tensor categories
        denseBase: 0x2e4659,
        svdCompressed: 0x3acbb8,
        svdEmissive: 0x103d36,
        moeRouter: 0xb57ee0,
        moeRouterEmissive: 0x2a143d,
        moeExpert: 0x9f68cb,
        mlaLatent: 0xe6a848,
        mlaEmissive: 0x3d280b,
        slicegpt: 0x38bdf8,
        macroBox: 0x172d47,
        embedding: 0x10b981,
        outputHead: 0xf43f5e,

        // Edge wireframe outlines
        edgeDense: 0x4f748f,
        edgeSvd: 0x53ecd9,
        edgeMoe: 0xc895ea,
        edgeMla: 0xf6be68,
        edgeMacro: 0x38bdf8,
        edgeResidual: 0x60a5fa,

        // Token materials
        tokenBaseline: 0x9fd2ff,
        tokenCompressed: 0x6effeb,
        trailBaseline: 0x4a82b8,
        trailCompressed: 0x289e90,
    }
};

/**
 * Foundational 4 Transformer Model Specifications.
 * Defines dimensions, layer depths, head allocations, and baseline parameter baselines.
 */
export const MODEL_SPECS = {
    distilbert: { 
        name: 'DistilBERT-Base', 
        family: 'Encoder', 
        num_layers: 6, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 66301440,
        descr: '6 Layers · 768d · 12 Heads · Distilled Encoder',
        vocab_size: 30522,
    },
    roberta: { 
        name: 'RoBERTa-Base', 
        family: 'Encoder', 
        num_layers: 12, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 123933696,
        descr: '12 Layers · 768d · 12 Heads · Bidirectional Encoder',
        vocab_size: 50265,
    },
    gpt2: { 
        name: 'GPT-2-Small', 
        family: 'Causal Decoder', 
        num_layers: 12, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 124318464,
        descr: '12 Layers · 768d · Fused c_attn Conv1D · Causal LM',
        vocab_size: 50257,
    },
    qwen: { 
        name: 'Qwen2.5-0.5B', 
        family: 'GQA Decoder (SLM)', 
        num_layers: 24, 
        hidden_size: 896, 
        intermediate_size: 4864, 
        base_layer_params: 14909440, 
        base_total: 494008320,
        descr: '24 Layers · 896d · GQA (14Q:2KV) · SwiGLU MLP',
        vocab_size: 151936,
    }
};

/**
 * Exact vocabulary sizes matching embedding matrix row dimensions.
 */
export const VOCAB_SIZES = {
    distilbert: 30522,
    roberta: 50265,
    gpt2: 50257,
    qwen: 151936,
};

/**
 * 5 Primary Compression Paradigms & Mathematical Formulations.
 */
export const PARADIGMS = {
    svd: {
        id: 'svd',
        name: 'Low-Rank SVD & Factorization',
        icon: '⚡',
        description: 'Decomposes dense weight matrix W into low-rank factors U and V with rank bottleneck r.',
        math: 'W \\approx U \\cdot V, \\quad W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}, \\; U \\in \\mathbb{R}^{d_{\\text{out}} \\times r}, \\; V \\in \\mathbb{R}^{r \\times d_{\\text{in}}}',
    },
    moe: {
        id: 'moe',
        name: 'Mixture of Experts & Routing',
        icon: '🔀',
        description: 'Dynamically routes incoming token representations through top-k specialized expert networks.',
        math: 'y = \\sum_{i=1}^N G(x)_i \\cdot E_i(x), \\quad G(x) = \\text{Softmax}(\\text{TopK}(x \\cdot W_g))',
    },
    mla: {
        id: 'mla',
        name: 'Latent & KV-Cache Compression (MLA)',
        icon: '🧠',
        description: 'Compresses Key and Value states into a shared low-rank latent vector prior to attention.',
        math: 'c_t^{KV} = W^{DKV} x_t, \\quad k_t^C = W^{UK} c_t^{KV}, \\quad v_t^C = W^{UV} c_t^{KV}',
    },
    slicegpt: {
        id: 'slicegpt',
        name: 'Structured Pruning & Channel Slicing',
        icon: '✂️',
        description: 'Applies orthogonal rotations to concentrate feature variance, then prunes uninformative channels.',
        math: 'W_{\\text{slice}} = (W \\cdot Q)_{[:, :d_{\\text{keep}}]}, \\quad Q \\in \\mathbb{R}^{d \\times d}, \\; Q^T Q = I',
    },
    quant: {
        id: 'quant',
        name: 'Quantization & Mixed Precision',
        icon: '💎',
        description: 'Maps continuous FP32/FP16 weights to discrete INT4/INT8 integer representations with per-channel scales.',
        math: '\\hat{W} = s \\cdot \\text{clamp}\\left(\\left\\lfloor \\frac{W}{s} + \\frac{1}{2} \\right\\rceil, -2^{b-1}, 2^{b-1}-1\\right)',
    }
};

/**
 * Color palette hex integers for the 4 parallel MoE expert routes.
 */
export const EXPERT_HEX_COLORS = [
    0x53ecd9, // Cyan (Expert 0)
    0xc895ea, // Purple (Expert 1)
    0xf6be68, // Amber (Expert 2)
    0xf472b6  // Rose (Expert 3)
];

/**
 * Architectural Guided Tour Waypoints (8 Stages).
 */
export const TOUR_STEPS = [
    { y: 2,  dist: 20, label: '01 — Input Embeddings & Positional Encoding', desc: 'Converts discrete token IDs into dense hidden vectors combined with learned/rotary positional embeddings.', target: null },
    { y: 5,  dist: 18, label: '02 — Multi-Head Q · K · V Projections',      desc: 'Projects hidden states into Query, Key, and Value subspaces (or low-rank bottleneck factors in SVD/MLA).', target: 'q_lin' },
    { y: 8,  dist: 16, label: '03 — Attention Output & Head Merging',       desc: 'Calculates scaled dot-product attention scores and projects concatenated head states back to hidden dimension.', target: 'out_lin' },
    { y: 10, dist: 16, label: '04 — Residual Skip Connection #1',           desc: 'Adds raw input tokens directly to attention outputs with LayerNorm/RMSNorm to prevent gradient decay.', target: null },
    { y: 13, dist: 18, label: '05 — FFN Intermediate Expansion',            desc: 'Expands token dimensions by 4× (or routes tokens across top-k experts in MoE architectures).', target: 'lin1' },
    { y: 16, dist: 18, label: '06 — FFN Down Projection & Contraction',      desc: 'Projects non-linear activations back to original hidden width, consolidating learned token features.', target: 'lin2' },
    { y: 20, dist: 22, label: '07 — Final Norm & Language Model Head',      desc: 'Normalizes final layer representations and maps logits over the full vocabulary distribution.', target: null },
    { y: 12, dist: 35, label: '08 — Global Macro Architecture Overview',    desc: 'Full network topology showing parallel baseline vs compressed parameter pathways and residual highways.', target: null, fullFit: true },
];
