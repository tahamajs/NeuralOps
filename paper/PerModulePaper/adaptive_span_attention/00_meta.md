# Adaptive Span Attention Module (adaptive_span_attention)

**Registry Keys**: `adaptive_span`, `adaptive_span_attention`, `adaptive_span_inject`
**Source**: `Codes/framework/attentions/adaptive_span_attention.py` <!-- source: Codes/framework/attentions/adaptive_span_attention.py lines 32, 169-170 -->
**Type**: Dynamic Continuous Span Masking with Learnable Soft Ramp (Sukhbaatar et al., 2019)
**Base Class**: `AdaptiveSpanAttention(BaseAttention)` <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 169 -->

## Abstract
Standard full self-attention scales quadratically with sequence length $\mathcal{O}(L^2 d)$ and applies a uniform context window across all attention heads. However, individual attention heads specialize in vastly different contextual scales: some track local syntactic dependencies within a few tokens, while others require broader semantic reach. `adaptive_span_attention` implements continuous, differentiable per-head attention span masking using a soft ramp function $g_{z_h}(d) = \operatorname{clamp}\left(\frac{R + z_h - d}{R}, 0, 1\right)$ where $z_h = \sigma(\text{logit}_h) \cdot S_{\max}$. Evaluated across DistilBERT, RoBERTa, GPT-2, and Qwen 3.5, it provides dynamic context trimming while maintaining high downstream accuracy with LoRA recovery.
