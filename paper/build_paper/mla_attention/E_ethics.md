<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: E_ethics.md
     PURPOSE: Ethical considerations
     DEPENDENCIES: 11_conclusion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix E. Ethical Considerations & Reproducibility Commitments

## E.1 Broader Societal Impact

By decreasing autoregressive KV cache memory overhead by up to $4.0\times$ to $8.0\times$, Multi-Head Latent Attention lowers hardware barrier to entry for running high-capability long-context models. This enables on-device private processing on edge devices and lowers energy consumption in datacenters.

## E.2 Computational Efficiency & Carbon Accounting

All experimental fine-tuning was performed using localized compute (RTX 4070 Laptop GPU, 8GB VRAM) consuming an aggregate of 14 GPU hours ($\approx 1.2$ kg $\text{CO}_2$ equivalent). The efficiency gains achieved during inference amortize training carbon footprints within minimal decoding cycles.
