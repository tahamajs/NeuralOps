# 1. Introduction

## 1.1 Motivation
Transformer projection matrices exhibit substantial heterogeneity in singular value decay rates. Forcing a uniform rank across all layers wastes capacity in sensitive attention layers while under-compressing parameter-dense feed-forward projections.

## 1.2 Our Approach
`adaptive_ranks` dynamically computes the rank per layer using singular value cumulative energy thresholding combined with activation covariance preconditioning. It replaces target linear projections with `SVDLinear` layers, guaranteeing physical parameter and FLOP reductions.
