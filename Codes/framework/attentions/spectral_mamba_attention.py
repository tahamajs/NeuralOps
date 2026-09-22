"""
Spectral-Mamba Attention Module: Hybrid Attention-State Space Module with Spectral Channel Modulation.

Combines pretrained multi-head attention with a Spectral Mamba SSM branch via learned adaptive gating.

NOVELTY STATEMENT:
    SpectralMamba combines frequency-domain spectral modulation with low-rank state space
    projection. Our specific formulation features:
    1. Channel-wise discrete Fourier transform (FFT) sequence filtering using learned
       frequency-domain gating masks across frequency bins.
    2. Zero-initialized low-rank state space adaptation branch that is cleanly disabled at
       initialization via a softmax-dominant gate (bias=10.0, sigmoid ≈ 0.99995).
    3. Multi-architecture weight transfer supporting MHA, GQA, nested encoders, and causal decoders.
"""

from framework.attentions.spectral_mamba import (
    SpectralMambaAttention,
    SpectralMambaConfig,
    SpectralMamba,
    LoRALinear,
    inject_spectral_mamba,
)

__all__ = [
    "SpectralMambaAttention",
    "SpectralMambaConfig",
    "SpectralMamba",
    "LoRALinear",
    "inject_spectral_mamba",
]
