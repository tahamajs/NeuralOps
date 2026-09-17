# 1. Introduction

## 1.1 Motivation
Standard low-rank matrix decomposition algorithms impose a uniform rank $r$ across all layers in a transformer. However, empirical studies demonstrate that different layers exhibit markedly different spectral decay rates: lower layers often retain broader rank distributions, while higher layers concentrate variance in a narrow subspace. Enforcing a global static rank over-compresses sensitive layers while under-compressing redundant ones.

## 1.2 Our Approach
`adaptive_compressor` dynamically selects the compression rank for each individual linear projection by inspecting its singular value energy spectrum. The rank $r$ is determined such that the cumulative energy exceeds a target threshold $\tau$ (default 0.92), bounded by an adaptive floor $r_{\min}$. As an engineered subclass of `AdaptiveRanksCompressor`, it provides a stable entry point for compression sweeps across encoder and decoder backbones.
