/**
 * educational/explanations.js
 * ===========================
 * Rich architectural explanations, pedagogical notes, and mathematical text
 * for all transformer tensor nodes and sublayers.
 */

import { anyKey } from '../utils/math.js';

/**
 * Returns pedagogical HTML description and educational note for a selected node.
 * @param {object} node - Selected 3D node
 * @param {boolean} isBaseline - Whether node belongs to baseline (left) or compressed (right)
 * @returns {{ descHtml: string, noteHtml: string }}
 */
export function getEducationalExplanation(node, isBaseline = false) {
    const meta = node.metadata || {};
    const shape = meta.shape || [768, 768];
    const dOut = shape[0] || 768;
    const dIn = shape[1] || 768;

    if (node.isMacro) {
        return {
            descHtml: `<strong>Macro Layer Block:</strong> Consolidated enclosing volume for ${node.parent_layer_name}. Contains all constituent sub-tensors with combined footprint of ${meta.params} (${meta.storage}).`,
            noteHtml: `💡 <strong>Semantic Zooming:</strong> Zoom in (scroll up or click "+") to unpack this macro block into individual Self-Attention and Feed-Forward matrix tensors!`
        };
    }

    if (node.type === 'embedding' || node.sub_type === 'input_embedding') {
        return {
            descHtml: `<strong>Input Tokenization & Embeddings (Tokens In):</strong> Token IDs $[t_1, \\dots, t_S]$ are indexed into vocabulary embedding matrix $W_e \\in \\mathbb{R}^{${dOut} \\times ${dIn}}$. Positional vectors (or RoPE complex rotation matrices $\\mathbf{R}_{\\Theta, m}$) are added to impart sequence order.`,
            noteHtml: `💡 <strong>Tokens Ingestion:</strong> This emerald base plate is where discrete language tokens enter the continuous geometric manifold of the model.`
        };
    }

    if (node.type === 'residual_add' || node.sub_type === 'add_operator') {
        const isAttnAdd = (node.id || '').includes('attn');
        return {
            descHtml: isAttnAdd
                ? `<strong>Attention Skip Highway (+X_in Addition):</strong> Incoming tokens $X_{\\text{in}}$ bypass the attention sublayer through the skip connection and are added elementwise: $X_{\\text{mid}} = X_{\\text{in}} + \\text{Dropout}(\\text{Attn}(X_{\\text{in}}))$. Prevents vanishing gradients and preserves identity information.`
                : `<strong>FFN Skip Highway (+X_mid Addition):</strong> Intermediate tokens $X_{\\text{mid}}$ bypass the feed-forward network and are added elementwise: $X_{\\text{layer}} = X_{\\text{mid}} + \\text{Dropout}(\\text{FFN}(X_{\\text{mid}}))$. Combines contextual features with non-linear factual memory.`,
            noteHtml: isAttnAdd
                ? `💡 <strong>Highway Addition #1:</strong> Trace the curved lateral bypass cable — notice how raw token identities are injected directly into the contextual delta!`
                : `💡 <strong>Highway Addition #2:</strong> Trace the second lateral bypass cable — notice how the accumulated token state is merged with the FFN factual output!`
        };
    }

    if (node.type === 'output_head' || node.sub_type === 'lm_head') {
        return {
            descHtml: `<strong>Output LM Head & Next-Token Distribution (Logits Out):</strong> The final normalized token states $X^{(L)} \\in \\mathbb{R}^{S \\times ${dIn}}$ are projected across the vocabulary $\\mathbb{R}^{${dOut}}$ via unembedding matrix $W_{\\text{vocab}}^T$, yielding logits $\\hat{y} = \\text{Softmax}(\\text{Norm}(X^{(L)}) W_{\\text{vocab}}^T)$.`,
            noteHtml: `💡 <strong>Next-Token Sampling:</strong> This rose top slab is where continuous token representations are converted into real probability distributions over the dictionary!`
        };
    }

    if (node.type === 'svd') {
        const rank = meta.rank || 64;
        return {
            descHtml: `<strong>Low-Rank Factorization ($W \\approx U \\cdot V$):</strong> Decomposes dense linear map $\\mathbb{R}^{${dIn}} \\to \\mathbb{R}^{${dOut}}$ into projection factor $V \\in \\mathbb{R}^{${rank} \\times ${dIn}}$ and expansion factor $U \\in \\mathbb{R}^{${dOut} \\times ${rank}}$. Parameter count drops from $${(dIn * dOut).toLocaleString()}$ to $${((dIn + dOut) * rank).toLocaleString()}$.`,
            noteHtml: `💡 <strong>Physical SVD Neck:</strong> Look at the 3D geometry — notice how the $V$-factor is razor-thin along the rank axis ($r=${rank}$), physically forcing token representations to compress into the principal eigenspace!`
        };
    }

    if (node.type === 'moe') {
        if (node.sub_type === 'router') {
            return {
                descHtml: `<strong>MoE Gating Router:</strong> Computes routing logits $G(x) = \\text{Softmax}(\\text{TopK}(x \\cdot W_g))$. Dispatches each token selectively to the most specialized expert subnetwork.`,
                noteHtml: `💡 <strong>Sparse Computation:</strong> Only the top-1 expert is evaluated per token, keeping FLOPs low while multiplying total parameter capacity!`
            };
        }
        return {
            descHtml: `<strong>Sparse Expert Module:</strong> Independent sub-network specialized for a subset of semantic tokens. Evaluates active representations with shape $[${dOut} \\times ${dIn}]$.`,
            noteHtml: `💡 <strong>Parallel Experts:</strong> Look at the 4 horizontally spaced expert blocks — watch how the token stream branches into distinct colors as tokens enter their routed expert!`
        };
    }

    if (node.type === 'norm' || (node.id && (node.id.includes('norm') || node.id.includes('ln')))) {
        const isPre = (node.id || '').includes('pre') || (node.id || '').includes('1');
        return {
            descHtml: `<strong>${isPre ? 'Pre-Attention' : 'Post-Attention'} LayerNorm / RMSNorm (Uncompressed Layer):</strong> Stabilizes the token representation variance $\\bar{x} = \\frac{x - \\mu}{\\sigma} \\odot \\gamma + \\beta$. This layer preserves exact numerical dynamic range and is <em>deliberately kept uncompressed (Full FP16/BF16 Precision)</em> to prevent representation drift across deep transformer stacks.`,
            noteHtml: `💡 <strong>Numerical Anchor:</strong> Normalization layers account for less than 0.2% of total parameters, so compressing them offers almost zero memory gain while drastically harming training stability!`
        };
    }

    if (node.id && (node.id.includes('gate_proj') || node.id.includes('up_proj') || node.id.includes('down_proj') || (node.label && node.label.includes('SwiGLU')))) {
        return {
            descHtml: `<strong>SwiGLU Feed-Forward Network (FFN Memory Retrieval):</strong> Computes non-linear associative factual retrieval: $\\text{FFN}(x) = (x W_{\\text{gate}}) \\odot \\text{SiLU}(x W_{\\text{up}}) W_{\\text{down}}$. Projects $d_{\\text{model}} \\to d_{\\text{intermediate}}$ with dimension $[${dOut} \\times ${dIn}]$ and parameter footprint of ${meta.params || '4.35M'}.`,
            noteHtml: `💡 <strong>Factual Memory Core:</strong> While Attention mixes context between different tokens, the FFN acts as a key-value associative memory storing factual knowledge learned during pre-training!`
        };
    }

    if (node.id && (node.id.includes('ffn') || node.id.includes('mlp') || node.id.includes('intermediate') || node.id.includes('lin2'))) {
        return {
            descHtml: `<strong>Feed-Forward Expansion & Down-Projection (MLP Sublayer):</strong> Expands hidden states 4x into non-linear intermediate dimension $d_{\\text{ffn}} = ${dOut}$, applies activation (GELU/ReLU), and compresses back down to $d_{\\text{model}} = ${dIn}$. Stores factual key-value patterns.`,
            noteHtml: `💡 <strong>Knowledge Capacity:</strong> The FFN layer accounts for approximately 66% of a standard transformer's parameter count, making it a prime candidate for SVD, MoE, and pruning!`
        };
    }

    if (node.id && anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj', 'c_attn'])) {
        const isGqa = (dOut <= 256 && (node.id.includes('k_proj') || node.id.includes('v_proj')));
        return {
            descHtml: `<strong>Self-Attention Multi-Head Linear Projection:</strong> Projects token vectors from $d_{\\text{model}} = ${dIn}$ into subspace $d_{\\text{head}} = ${dOut}$. ${isGqa ? 'Uses Grouped-Query Attention (GQA) with shared KV heads to save 7x KV cache memory.' : 'Full multi-head representation.'}`,
            noteHtml: `💡 <strong>Attention Routing:</strong> Notice the distinct vertical head partition lines across the box — each segment represents an independent attention head computing distinct semantic queries!`
        };
    }

    // Default Dense Projection
    return {
        descHtml: `<strong>Dense Linear Projection (Uncompressed Baseline):</strong> Full-rank transformation matrix mapping representations from $\\mathbb{R}^{${dIn}}$ to $\\mathbb{R}^{${dOut}}$ with standard $O(N^2)$ quadratic computational complexity.`,
        noteHtml: `💡 <strong>Dense Monolith:</strong> Notice the thick solid geometry — this uncompressed weight matrix stores all ${meta.params || 'weights'} in full uncompressed precision without bottlenecking.`
    };
}
