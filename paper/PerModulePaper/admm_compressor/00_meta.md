# ADMM Joint Pruning and Quantization Compressor (admm_compressor)

**Registry Keys**: `admm`, `admm_compressor` <!-- source: Codes/framework/compression_engine/admm_compressor.py lines 43-44 -->
**Source**: `Codes/framework/compression_engine/admm_compressor.py` (423 lines)
**Type**: Alternating Direction Method of Multipliers for Structured Pruning and Quantization
**Base Class**: `ADMMCompressor(BaseCompressor)` <!-- source: Codes/framework/compression_engine/admm_compressor.py line 45 -->

## Abstract
`admm_compressor` formulates weight compression as an optimization problem under non-convex sparsity and discrete quantization constraints using the Alternating Direction Method of Multipliers (ADMM). Operating in strict FP32 precision, it optimizes primal weight tensors $W$, auxiliary sparse/quantized projections $Z$ and $Q$, and dual multipliers $U_1, U_2$ with a monotonic error tracker that preserves the best iterate across optimization steps. Across DistilBERT, RoBERTa, GPT-2, and Qwen 3.5, it provides robust convergence, retaining 90.37% accuracy on DistilBERT and 88.42% on Qwen 3.5.
