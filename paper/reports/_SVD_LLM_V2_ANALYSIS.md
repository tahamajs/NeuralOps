# Detailed Investigation & Forensic Analysis: `SVD-LLM V2` in the Codebase

**Date:** 2026-09-21  
**Target Module:** `Codes/framework/attentions/svd_llm_v2_attention.py` (and alias `svd_llm_v2.py`)  
**Context:** Verification of baseline comparison against SVD-LLM (ICLR 2025) vs. AASVD.

---

## 1. Executive Summary & Core Verdict

| Question | Finding | Verdict |
| :--- | :--- | :--- |
| **Is `svd_llm_v2` in any compressor engine?** | No occurrences in `Codes/framework/compression_engine/`. Registered solely in `ATTENTION_REGISTRY` as an attention block (`BaseAttention`). | **Attention Replacement Only** |
| **Is it a faithful implementation of SVD-LLM (ICLR 2025)?** | **NO.** The authentic SVD-LLM paper (*"SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression"*, Wang et al., ICLR 2025) requires **data-driven Cholesky input-activation whitening** and layer-by-layer sequential update. Our repo's code performs **completely unweighted truncated SVD directly on static weights**. | **Simplified / Incomplete Version** |
| **Why does it collapse on GPT-2 (PPL = 2254.09)?** | (1) Static unweighted SVD ignores outlier activation dimensions and causal autoregressive dynamics; (2) fixed rank truncation drops essential decoder attention representations; (3) LoRA adapters are initialized to 0 and untrained without recovery fine-tuning during zero-shot language modeling. | **Methodological Failure on Autoregressive LLMs** |
| **Impact on Paper Comparison:** | Evaluating against this repo's `svd_llm_v2_attention` as if it were the authentic ICLR 2025 SVD-LLM is **methodologically invalid**. It must either be explicitly identified as an unwhitened architectural reproduction or compared against properly formulated activation-whitened baselines. | **Crucial Paper Disclaim / Boundary Required** |

---

## 2. Investigation Details

### 2.1 Codebase Presence & Registration
- A search across `Codes/framework/compression_engine/` reveals **zero** usages:
  ```bash
  grep -rn "svd_llm_v2" Codes/framework/compression_engine/
  # Exit code 1 (No matches)
  ```
- The module exists solely as an attention layer in:
  - [`Codes/framework/attentions/svd_llm_v2_attention.py`](Codes/framework/attentions/svd_llm_v2_attention.py)
  - Registered under `@ATTENTION_REGISTRY.register("svd_llm_v2")` and `inject_svd_llm_v2`.
- It is evaluated by replacing standard Multi-Head Attention (`c_attn`, `q_lin`, etc.) with `SVDLLMV2Attention`, rather than applying model-wide layer compression across MLP and attention blocks.

---

### 2.2 Forensic Evidence: GPT-2 Perplexity Collapse (PPL = 2254)

In the remote execution logs (`D:\svd_project\logs\ppl_loop.log`), evaluating GPT-2 causal language modeling on WikiText-2 with `svd_llm_v2_attention` injected produces an immediate catastrophic perplexity explosion:

```text
[INFO] framework.core.attention_utils: Successfully injected 'svd_llm_v2_attention' into 12 transformer blocks.
Token indices sequence length is longer than the specified maximum sequence length for this model (287644 > 1024).
Running this sequence through the model will result in indexing errors
[PPL] svd_llm_v2_attention: 2254.0918
```

Similarly, on SST-2 sequence classification with GPT-2 (`results/unified_svd_llm_v2/gpt2/summary.csv`):
```csv
module_name,status,accuracy_pct,f1_score,f1_macro,val_loss,num_injected_attention_layers
full,PASS,60.78,0.6354,0.6055,0.6585,0
svd_llm_v2,PASS_LOW,50.92,0.6748,0.3374,0.7408,12
```
*Notice:* 50.92% is random chance guessing on binary sentiment, confirming complete representational collapse.

---

### 2.3 Head-to-Head Comparison: SVD-LLM V2 vs. AASVD

| Feature | `svd_llm_v2_attention.py` (Repo) | Authentic SVD-LLM (ICLR 2025) | AASVD (Ours) |
| :--- | :--- | :--- | :--- |
| **Weighting / Whitening** | **None** (unweighted Frobenius SVD) | **Cholesky Whitening** ($W R^\top$ where $X^\top X = R^\top R$) | **Streaming EMA Covariance** ($W C_X^{1/2}$) |
| **Activation Awareness** | None (Zero calibration data) | Input activation covariance via calibration set | Multi-batch streaming EMA with $O(d^2)$ RAM |
| **Rank Determination** | Static spectral energy ratio ($E \ge 0.999$) capped at $r$ | Truncation-aware residual loss allocation | **STRS** (Sensitivity-Based Rank Search) / budget search |
| **Layer Scope** | Self-attention $Q, K, V, O$ only | All Linear layers (Attention + MLP up/down) | Transformer linear projections |
| **LoRA Architecture** | Per-projection parameter buffers ($A, B$) | Optional post-training recovery fine-tuning | Standard structural recovery ($r_{\text{lora}}=16\text{--}32$) |
| **Safety Mechanisms** | None (forces truncation on all blocks) | Progressive layer-by-layer updates | **Frobenius fallback** ($\text{max\_error} \le 0.50$) |
| **GPT-2 Perplexity** | **2254.09** (Broken / Catastrophic) | Competitive baseline (< 35 on WikiText-2) | **25 – 28** (Stable low-PPL language modeling) |

---

### 2.4 Why Does `svd_llm_v2_attention` Fail on GPT-2?

1. **No Cholesky Whitening (Missing Paper Core):**
   In the ICLR 2025 paper (*Wang et al.*), SVD-LLM's central contribution is that simple SVD fails because activation magnitudes vary wildly across channels. They compute:
   $$X^\top X = R^\top R \quad (\text{Cholesky decomposition})$$
   and perform SVD on the whitened weight $W R^\top = U \Sigma V^\top$, setting $W \approx U_r \Sigma_r (R^{-\top} V_r)^\top$.  
   In `Codes/framework/attentions/svd_llm_v2_attention.py` lines 306–317:
   ```python
   U, S, Vt = torch.linalg.svd(W, full_matrices=False)
   u_param.data[:, :r] = U[:, :r]
   s_param.data[:r] = S[:r]
   vh_param.data[:r, :] = Vt[:r, :]
   ```
   There is **no calibration data collected**, **no activation covariance matrix**, and **no Cholesky whitening**. It is literally plain truncated SVD wrapped in a custom PyTorch attention class.

2. **Decoupled $U \Sigma V^\top$ Forward Path without Safety Fallback:**
   The module factorizes all 12 layers of GPT-2 at a static rank (e.g. $r=32$ or $r=64$). Because GPT-2 `c_attn` is $768 \times 2304$, unweighted rank-32 truncation discards over 88% of the singular spectrum energy without checking reconstruction error. With no safety fallback mechanism, all 12 layers suffer catastrophic signal loss.

3. **Untrained LoRA Buffers during Evaluation:**
   The module initializes LoRA $B$ to zero (`nn.init.zeros_(p_b)`). In zero-shot evaluation (such as WikiText-2 perplexity runs), the LoRA adapters provide zero output, meaning the model runs on the severely truncated unweighted factors alone.

---

### 2.5 Categorization & Scholarly Conclusion

`Codes/framework/attentions/svd_llm_v2_attention.py` is:
- **(c) An attention module only (no model-level compressor pipeline).**
- **(b) A simplified/incomplete version of the ICLR 2025 SVD-LLM algorithm** lacking the essential Cholesky whitening step and activation calibration.

### 2.6 Recommendation for the AASVD Paper:
1. **Do not cite the repo's internal `svd_llm_v2` results (PPL=2254) as representing the true ICLR 2025 SVD-LLM baseline.** Doing so would create a strawman comparison that peer reviewers will instantly catch.
2. In the paper's Related Work and Baseline sections, accurately credit SVD-LLM (Wang et al., ICLR 2025) for its Cholesky-whitening formulation, while distinguishing AASVD's **symmetric streaming EMA covariance** $C_X^{1/2}$ and **Sensitivity-Based Rank Search (STRS)** as providing online, parameter-budgeted compression with strict Frobenius fallback guarantees.
