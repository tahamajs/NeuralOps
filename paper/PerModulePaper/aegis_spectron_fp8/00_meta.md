# Aegis-Spectron-FP8 Hybrid Attention Module (aegis_spectron_fp8)

**Registry Keys**: `inject_aegis_spectron_fp8`, `aegis_spectron_fp8_attn`, `aegis_spectron_fp8_inject`, `aegis_spectron_fp8` <!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 29 -->
**Source**: `Codes/framework/attentions/aegis_spectron_fp8.py` (1023 lines)
**Type**: Multi-Branch Hybrid Attention with FP8 Simulation and Spectral Entropy Routing
**Base Class**: `AegisSpectronFP8Attention(BaseAttention)`

## Abstract
`aegis_spectron_fp8` integrates multi-branch local/global attention with Spectron spectral entropy routing and simulated FP8 quantization (E4M3 and E5M2) with dynamic scaling. Projections utilize `FactorizedLinear` layers to enforce true physical parameter reductions. Evaluated on SST-2, it achieves 79.59% accuracy on DistilBERT with an 18.22% parameter reduction (saving 12.2M parameters) and peak VRAM of only 290.3 MB.
