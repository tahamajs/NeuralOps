# Factorised Matrix State Attention Audit & Hardening Report

**Module:** `Codes/framework/attentions/factorised_matrix_state.py`  
**Registry Names:** `fms`, `factorised_matrix_state`  
**Test Suite:** `Codes/framework/tests/test_fms.py`  
**Date:** September 22, 2026  

---

## 1. Executive Summary

The Factorised Matrix State Attention module had several critical mathematical and structural issues:
1. **False $O(L)$ Linear Complexity Claim:** The previous implementation constructed an $L \times L$ Toeplitz decay matrix $D \in \mathbb{R}^{L \times L}$ and computed full sequence-by-sequence pairwise attention matrix scores:
   $$\text{scores} = (Q_f K_f^\top) \odot D \in \mathbb{R}^{B \times H \times L \times L}$$
   This was an $O(L^2)$ time and $O(L^2)$ memory materialization, invalidating the claim of true linear memory complexity.
2. **Zero-Initialization Saddle Point in $L$ and $R$:** Both $L \in \mathbb{R}^{D \times r}$ and $R \in \mathbb{R}^{D \times r}$ were initialized to all zeros. Since the factorized output projection computes $(y L) R^\top$, the gradient with respect to $L$ contains $R = 0$, and the gradient with respect to $R$ contains $L = 0$. Consequently, $\nabla_L \mathcal{L} = 0$ and $\nabla_R \mathcal{L} = 0$, trapping both projection matrices in a dead saddle point under first-order gradient descent.
3. **Missing `logger` Import:** `logger.warning` was invoked without `logger` or `logging` being defined, leading to `NameError` whenever warnings were logged.
4. **Odd Head Dimension Crash in `_rotate_half`:** The `_rotate_half` helper split the head dimension as `d // 2`, causing index errors if $d$ is odd.
5. **Silent Weight Fallback:** If mandatory projection matrices ($Q, K, V$) were missing during `copy_weights`, the module proceeded silently with randomly initialized weights rather than raising an informative `RuntimeError`.

---

## 2. Remediations Applied

1. **Chunked Associative Linear Scan:**
   - Implemented a chunked linear recurrence algorithm with intra-chunk parallel attention and inter-chunk running state accumulation.
   - Maintains recurrent associative matrix state $S_t \in \mathbb{R}^{B \times H \times d \times d}$ and normalizer state $z_t \in \mathbb{R}^{B \times H \times d \times 1}$ across chunks.
   - For long sequences, time complexity is strictly $O(L \cdot d^2)$ and memory is $O(C^2 + d^2)$ where $C$ is chunk size ($64$), delivering true $O(L)$ linear scaling without $O(L^2)$ matrix explosion.
2. **Symmetry Breaking in $L$ and $R$:**
   - Initialized $L$ with Kaiming uniform initialization ($a = \sqrt{5}$) and $R$ with exact zeros:
     $$\Delta W = L R^\top = L \cdot 0 = 0$$
   - The initial perturbation remains strictly zero at initialization (preserving the pretrained weights transfer in $W_{\text{res}}$), but $\nabla_R \mathcal{L} = (y L)^\top (\nabla_{\text{out}} \mathcal{L}) \neq 0$, breaking symmetry and allowing immediate learning.
3. **Odd Dimension Rotary Safety:**
   - Updated `_rotate_half` to rotate the even prefix and concatenate any trailing odd coordinate unchanged.
4. **Logging & Strict Validation:**
   - Imported standard `logging` and configured `logger = logging.getLogger(__name__)`.
   - Raised explicit `RuntimeError` if mandatory projections are missing from the source module.

---

## 3. Empirical Verification & Test Results

The test suite `Codes/framework/tests/test_fms.py` passed all 11 test cases:
1. `test_basic_forward`: Verified bidirectional attention with ELU+1 feature map.
2. `test_causal_forward`: Verified unidirectional causal scan for autoregressive models.
3. `test_chunked_scan_equivalence`: Verified mathematical equivalence between chunked scan and parallel scan ($\Delta < 10^{-4}$).
4. `test_long_sequence_linear_scaling`: Verified execution on long sequences ($L=256$) with constant memory footprint.
5. `test_distilbert_weight_copy`: Verified exact projection weight copying from DistilBERT.
6. `test_roberta_weight_copy_and_nested_bypass`: Verified RoBERTa nested encoder bypass.
7. `test_gpt2_weight_copy`: Verified fused `c_attn` projection extraction and causal recurrence.
8. `test_qwen_gqa_and_rope`: Verified GQA head expansion and RoPE positional embeddings.
9. `test_missing_weights_raises`: Verified `RuntimeError` raised on missing weights.
10. `test_odd_head_dimension_rotate_half`: Verified odd head dimension rotation ($d=17$).
11. `test_lr_symmetry_breaking`: Verified non-zero norm on $L$ and zero norm on $R$.

**Pass Rate:** 11/11 tests passed (100%).
