# Adaptive Transformer Compressor (adaptive_compressor)

**Registry Keys**: `adaptive_compressor`, `adaptive_svd_factory`, `adaptive_compressor_factory`
**Source**: `Codes/framework/compression_engine/adaptive_compressor.py` <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 19 -->
**Type**: Dynamic Energy-Based SVD Linear Compressor (Unified Alias Wrapper)
**Base Class**: `AdaptiveTransformerCompressor(AdaptiveRanksCompressor)` <!-- source: Codes/framework/compression_engine/adaptive_compressor.py line 20 -->

## Abstract
`adaptive_compressor` provides the unified high-level interface for adaptive singular value decomposition in transformer architectures. Delegating directly to the production-hardened `AdaptiveRanksCompressor`, it resolves namespace and registry collisions while exposing dynamic cumulative energy thresholding $\tau = 0.92$, singular value energy spectrum allocation, and automatic layer-by-layer rank assignment. Across DistilBERT, RoBERTa, and GPT-2, it ensures guaranteed parameter compression and seamless backward compatibility.
