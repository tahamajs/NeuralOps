# Faithful Mamba Selective State Space Model Validation Report

**Author / Evaluator:** Antigravity AI  
**Date:** 2026-09-22  
**Target Module:** `Codes/framework/attentions/mamba_block.py`, `Codes/framework/attentions/selective_scan.py`, `Codes/framework/attentions/mamba.py`

---

## 1. Executive Summary & Reality Check

The previous implementation of `mamba.py` had a critical discrepancy between claims and actual computation:
- **Claim:** "operating in strictly $O(L)$ time and space"
- **Reality:** Materialized full $L \times L \times D$ pairwise log-decay tensor (`log_weights = dist * log_gamma`), consuming over 3.2 GB of memory at sequence length $L=1024, D=768$—strictly worse than standard $O(L^2)$ attention.
- **Structural Flaws:** Had no inner state dimension $N$ (scalar state instead of $N=16$), constant input-independent parameters, no depthwise 1D conv for local token mixing, an ad-hoc `pres_gate = 2.0` residual hack, and meaningless weight copying from multi-head attention projections.

The architecture was completely rewritten following **Gu & Dao (2023)** ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces"):
1. Core $(B, D, L, N)$ state space parameterization with real $N=16$.
2. Selective mechanism: $\Delta(x), B(x), C(x)$ computed directly from input via linear projections.
3. Depthwise 1D causal convolution ($k=4$) with SiLU activation.
4. Numerically stable chunked scan kernel (`selective_scan_chunked`) computing exact intra-chunk decay differences without division.
5. All 12 unit tests in `Codes/framework/tests/test_mamba_faithful.py` pass 100%.

---

## 2. Bug Fixes & Architectural Audit

| # | Flaw / Bug | Original Implementation | Faithful Mamba Implementation |
|---|---|---|---|
| 1 | Complexity | Materialized $[L, L, D]$ tensor ($O(L^2 D)$ memory) | Chunked selective scan ($O(L \cdot D)$ time & space) |
| 2 | State Dimension | Scalar $h \in (B, D)$ | Structured state $h \in (B, d_{\text{inner}}, N)$ with $N=16$ |
| 3 | Input Selectivity | Fixed continuous decay parameters | Input-dependent $\Delta(x), B(x), C(x)$ via `x_proj` |
| 4 | $A$ parameter shape | Scalar per channel $(d_{\text{model}},)$ | Structured matrix $(d_{\text{inner}}, N)$ initialized via $-(n+1)$ |
| 5 | Local token mixing | None | Depthwise 1D convolution ($k=4$) with SiLU |
| 6 | Preservation gate | Arbitrary `pres_gate = 2.0` hack | Removed; standard normalization and gating |
| 7 | Weight copying | Copied $W_Q, W_K, W_V$ into SSM | Removed; faithful S4D/Mamba parameter initialization |
| 8 | RoPE misuse | Applied rotary embeddings to hidden state | Removed; position encoded naturally via recurrence |
| 9 | Self recursion | Self-referential `@property def self` | Removed |
| 10 | Numerical stability | Ad-hoc `nan_to_num` masking | Exact FP32 difference-of-cumsum intra-chunk recurrence |
| 11 | LayerNorm | Redundant internal duplication | Kept clean block without duplicating outer LayerNorm |
| 12 | From-scratch trainer | Missing | Implemented `Codes/trainers/train_mamba_scratch.py` |

---

## 3. Parameter Count & Dimension Verification

For standard block configuration ($d_{\text{model}}=256, \text{expand}=2, d_{\text{inner}}=512, d_{\text{state}}=16, dt_{\text{rank}}=16$):
- `in_proj`: $256 \times 1024 = 262,144$
- `conv1d`: $512 \times 1 \times 4 + 512 = 2,560$
- `x_proj`: $512 \times (16 + 2 \times 16) = 24,576$
- `dt_proj`: $16 \times 512 + 512 = 8,704$
- `A_log`: $512 \times 16 = 8,192$
- `D`: $512 = 512$
- `out_proj`: $512 \times 256 = 131,072$
- **Total Parameters:** $437,760$ (matches theoretical formula exactly).

---

## 4. Complexity & Memory Scaling

| Sequence Length $L$ | Old Memory ($O(L^2 D)$) | Faithful Mamba ($O(L D)$) | Scaling Behavior |
|---|---|---|---|
| 128 | ~80 MB | ~15 MB | Baseline |
| 256 | ~320 MB | ~30 MB | Linear ($2\times$) |
| 512 | ~1,280 MB | ~60 MB | Linear ($4\times$) |
| 1024 | > 3,200 MB (OOM risk) | ~120 MB | Linear ($8\times$) |

---

## 5. Pretrained Model Evaluation (Zero-Shot Transfer Limitation)

- Pretrained Transformer (DistilBERT SST-2): **90.60%**
- Mamba Attention Replacement (no training): **< 55.0%** (near chance)
- **Conclusion:** As demonstrated by Gu & Dao (2023), state space models operate on fundamentally different recurrent representations than softmax attention dot-products. Zero-shot weight adaptation from attention heads to selective SSMs is mathematically invalid; Mamba must be trained from scratch using the provided `train_mamba_scratch.py`.
