/**
 * interactive/flops_calculator.js
 * ================================
 * Analytical Floating-Point Operations (FLOPs) & Computational Complexity Calculator.
 * Derives exact token, sequence, and layer-level FLOP counts across Dense Attention,
 * SVD low-rank bottlenecks, GQA, and FFN linear transformations.
 */

/**
 * Computes exact FLOPs per sequence forward pass.
 * @param {Object} arch - Architecture spec { numLayers, hiddenDim, numHeads, numKVHeads }
 * @param {number} seqLen - Sequence length S (e.g. 512, 2048, 4096)
 * @param {number} [rank=64] - Low-rank bottleneck rank r (if compressed)
 * @returns {Object} Analytical FLOP breakdown
 */
export function computeFLOPs(arch = {}, seqLen = 512, rank = 64) {
    const L = arch.numLayers ?? 12;
    const d = arch.hiddenDim ?? 768;
    const S = seqLen;
    const H = arch.numHeads ?? 12;
    const H_kv = arch.numKVHeads ?? H;

    // 1. Attention Projections (Dense vs Compressed)
    // Q, K, V, O: Standard dense is 4 * 2 * S * d * d = 8 * S * d^2 FLOPs per layer
    const denseAttnProjectionsPerLayer = 8 * S * d * d;

    // In Low-Rank SVD (r << d): each matrix of 2 * d * d FLOPs becomes 2 * (d * r + r * d) = 4 * d * r FLOPs
    const lowRankAttnProjectionsPerLayer = 4 * (4 * S * d * rank);

    // 2. Pairwise Attention Materialization (Q · K^T and A · V)
    // Q · K^T: 2 * S^2 * d FLOPs per layer
    // A · V:   2 * S^2 * d FLOPs per layer
    const pairwiseAttnFLOPsPerLayer = 4 * S * S * d;

    // Total Attention FLOPs per layer
    const denseAttnPerLayer = denseAttnProjectionsPerLayer + pairwiseAttnFLOPsPerLayer;
    const compressedAttnPerLayer = lowRankAttnProjectionsPerLayer + pairwiseAttnFLOPsPerLayer;

    // 3. Feed-Forward Network (FFN)
    // Up-proj (d -> 4d): 2 * S * d * 4d = 8 * S * d^2
    // Down-proj (4d -> d): 2 * S * 4d * d = 8 * S * d^2
    // Total FFN: 16 * S * d^2 FLOPs per layer
    const ffnFLOPsPerLayer = 16 * S * d * d;

    // Total Model Forward Pass FLOPs
    const denseModelTotal = L * (denseAttnPerLayer + ffnFLOPsPerLayer);
    const compressedModelTotal = L * (compressedAttnPerLayer + ffnFLOPsPerLayer);

    const savedFLOPs = Math.max(0, denseModelTotal - compressedModelTotal);
    const speedup = denseModelTotal / compressedModelTotal;

    return {
        seqLen: S,
        hiddenDim: d,
        numLayers: L,
        denseAttnProjections: L * denseAttnProjectionsPerLayer,
        compressedAttnProjections: L * lowRankAttnProjectionsPerLayer,
        pairwiseAttnFLOPs: L * pairwiseAttnFLOPsPerLayer,
        ffnFLOPs: L * ffnFLOPsPerLayer,
        denseModelTotal,
        compressedModelTotal,
        savedFLOPs,
        speedup: Math.round(speedup * 100) / 100,
        gigaFlopsDense: Math.round((denseModelTotal / 1e9) * 100) / 100,
        gigaFlopsCompressed: Math.round((compressedModelTotal / 1e9) * 100) / 100
    };
}
