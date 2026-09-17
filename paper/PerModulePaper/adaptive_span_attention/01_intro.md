# 1. Introduction

## 1.1 Motivation
Standard multi-head attention computes attention weights across the entire sequence length for all $H$ heads. This is computationally redundant: empirical analysis demonstrates that many attention heads operate primarily over localized contexts (e.g., adjacent words, local phrase boundaries). Enforcing a full attention span on every head wastes compute and memory bandwidth.

## 1.2 Our Approach
Following Sukhbaatar et al. (2019), `adaptive_span_attention` equips each attention head with an independent, learnable span parameter $z_h$. A continuous masking function smoothly ramps attention weights to zero for relative token distances $d > z_h$. Through gradient descent, heads dynamically contract their effective spans, reducing context redundancy.
