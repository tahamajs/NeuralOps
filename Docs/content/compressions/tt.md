# Tensor-Train (TT) Matrix Factorization & Linear Compression

> [!WARNING]
> **DEPRECATION & BENCHMARK QUEUE REMOVAL NOTICE**
> 
> As of September 2026, **Tensor-Train (TT) compression has been formally deprecated and removed from the active automated benchmark queue** (`run_all_benchmarks_queue.py`).
>
> The module implementations (`TTLinearLayer` and `TTCompressor`) and empirical runners have been preserved under `Codes/framework/compression_engine/` and `Codes/experimental/runners/` strictly for historical baseline comparisons and archival research.
>
> **Summary of Reasons for Deprecation:**
> 1. **Inherent Dimensional Disadvantage on 2D Linear Layers**: Factorizing a 2D matrix into a 4-mode tensor product artificializes the matrix spectrum and scrambles row-column correlations, yielding far worse rank-compression trade-offs than 2D Singular Value Decomposition (SVD).
> 2. **Negligible Parameter Savings**: On realistic pretrained weights (GPT-2, RoBERTa), TT requires ranks $R_{\text{TT}} \ge 226 \text{--} 400$ to reach relative error $<0.30$, producing less than **0.29% net model parameter reduction** (vs. 66% with standard SVD).
> 3. **Zero FLOPs Reduction & GPU Kernel Overhead**: Tensor contraction via naive einsum runs **17x slower** on GPU due to uncoalesced memory access. Pre-contracting cores via `tensordot` restores latency to 1.1x–1.6x baseline, but materializes the full dense matrix $W$, delivering **0% FLOPs reduction**.
> 4. **Rank Runaway Vulnerability**: Unconstrained rank-adaptation loops attempt to lower reconstruction error by increasing ranks, paradoxically inflating module parameter count to **1.40x baseline (+3.5% model growth)** if not strictly clamped by mathematical upper bounds.

---

## 1. Executive Summary & Context

Tensor-Train (TT) decomposition (Oseledets, 2011; Novikov et al., 2015) factorizes high-dimensional tensors into a linear chain of 3-way core tensors. When applied to neural network linear layers $y = x W^T$, a 2D weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is factorized by reshaping the input and output dimensions:
$$
d_{\text{in}} = \prod_{k=1}^d p_k, \quad d_{\text{out}} = \prod_{k=1}^d q_k
$$
and viewing $W$ as a tensor of order $2d$. In typical transformer implementations (e.g. $d=2$), $W$ is mapped to a 4D tensor $\mathcal{W} \in \mathbb{R}^{p_1 \times p_2 \times q_1 \times q_2}$ and decomposed into two 4D TT-cores $G_1 \in \mathbb{R}^{1 \times p_1 \times q_1 \times r}$ and $G_2 \in \mathbb{R}^{r \times p_2 \times q_2 \times 1}$.

While TT provides exponential parameter reduction for genuinely high-order tensors (e.g. video processing, multi-dimensional PDE solving, or high-order embedding tables where $d \ge 4$), extensive benchmarking on transformer linear layers demonstrates that **TT is fundamentally ill-suited for compressing 2D linear weight matrices**.

---

## 2. Theoretical Analysis: Why TT Underperforms SVD on 2D Matrices

### 2.1 Mode Entanglement & Singular Value Flattening

In standard Singular Value Decomposition (SVD), the matrix $W \in \mathbb{R}^{M \times N}$ is factored as:
$$
W = U \Sigma V^T, \quad \Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(M, N)})
$$
By the Eckart-Young-Mirsky theorem, truncated SVD at rank $r$ is the **strictly optimal** rank-$r$ approximation with respect to both Frobenius and spectral norms:
$$
\min_{\operatorname{rank}(A) \le r} \|W - A\|_F = \sqrt{\sum_{i=r+1}^{\min(M, N)} \sigma_i^2}
$$

In contrast, TT matrix factorization splits $M = q_1 q_2$ and $N = p_1 p_2$, interleaving row and column indices:
$$
W_{(i_1, i_2), (j_1, j_2)} \longleftrightarrow \mathcal{W}_{(i_1, j_1), (i_2, j_2)}
$$
TT-SVD then flattens $\mathcal{W}$ into an unfolded matrix of size $(q_1 p_1) \times (q_2 p_2)$ and performs SVD across this combined mode partition.

**The Fundamental Flaw:**
- Natural transformer weight matrices possess strong directional alignment between inputs and outputs (e.g., query-key alignments, feed-forward projections).
- Interleaving $(i_1, j_1)$ and $(i_2, j_2)$ breaks the natural subspace alignment of the linear operator.
- The singular value spectrum of the unfolded tensor $\mathcal{W}_{[(i_1, j_1); (i_2, j_2)]}$ is artificially flattened compared to $W$.
- Because singular values decay much slower after mode scrambling, retaining a low relative error $\epsilon \le 0.30$ requires an order of magnitude higher rank $r$ than standard 2D SVD.

### 2.2 Parameter Scaling Comparison

For a typical transformer linear layer (e.g. GPT-2 small attention projection $768 \times 768$ or FFN projection $768 \times 3072$):

| Matrix Dimensions $(d_{\text{in}} \times d_{\text{out}})$ | Mode Splitting $(p_1 \times p_2, q_1 \times q_2)$ | SVD Parameters ($r=128$) | TT Parameters ($R_{\text{TT}}=16$) | TT Parameters ($R_{\text{TT}}=128$) | TT Upper Bound Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention Proj** ($768 \times 768$) | $(24 \times 32, 24 \times 32)$ | $128 \times (768 + 768) = 196,608$ (66.7% saving) | $16 \times (768 + 768) = 24,576$ (Error > 80%) | $128 \times (768 + 768) = 196,608$ (Error > 45%) | $r_{\text{max}} = 331$ ($508,416$ params) |
| **FFN Intermediate** ($768 \times 3072$) | $(24 \times 32, 48 \times 64)$ | $128 \times (768 + 3072) = 491,520$ (79.2% saving) | $16 \times (1152 + 2048) = 51,200$ (Error > 90%) | $128 \times (1152 + 2048) = 409,600$ (Error > 55%) | $r_{\text{max}} = 663$ ($2,121,600$ params) |

When $R_{\text{TT}}$ is increased to achieve an acceptable reconstruction error ($\le 0.30$):
- $R_{\text{TT}}$ must reach $\ge 226 \text{--} 400$.
- At these ranks, $P_{\text{TT}} = R_{\text{TT}} (p_1 q_1 + p_2 q_2)$ approaches or exceeds $d_{\text{in}} \cdot d_{\text{out}}$, nullifying all compression benefits.

---

## 3. Empirical Findings on Pretrained Weights

Rigorous evaluation was conducted on pretrained GPT-2 weights across multiple linear projection layers (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`):

### 3.1 Layer-by-Layer Behavior (`max_error = 0.30`)
1. **Attention Projections (`c_attn`, $768 \times 2304$)**:
   - Initial TT rank $r=16$ yielded relative error $> 0.78$.
   - Adaptive retries escalated rank towards the theoretical ceiling ($r \approx 300$).
   - Failed error threshold and cleanly fell back to original dense layer.
2. **MLP Projections (`c_fc`, $768 \times 3072$ and `c_proj`, $3072 \times 768$)**:
   - Singular values of FFN layers decay much more uniformly across modes.
   - At rank ceiling, relative error remained at $0.48 \text{--} 0.54$.
   - 100% of MLP layers fell back to uncompressed weights.
3. **Net Result**:
   - Only 2 out of 5 layers compressed even when relaxing threshold to `max_error = 0.50`.
   - Total model parameter reduction: **0.29%**.

### 3.2 Rank Runaway & Parameter Inflation Bug
In early unconstrained implementations, when `max_error` was not satisfied, the rank increment formula:
$$
\Delta r = \max\left(4, \frac{\min(d_{\text{in}}, d_{\text{out}}) - r_{\text{curr}}}{\text{retries}}\right)
$$
pushed $r$ up to $\min(d_{\text{in}}, d_{\text{out}}) = 768$. 
At $r=539$, TT parameters equaled **827,968** for a layer that originally had only **590,592** parameters (1.4019x parameter expansion). On full GPT-2, this caused the model to grow from 126.8M to 131.2M parameters (+3.5% growth).

While strict mathematical bounding (`_max_rank_for_compression`) fixed the parameter expansion, it revealed that TT could not compress the majority of layers without catastrophic quality degradation.

---

## 4. Hardware Execution & Roofline Analysis

Linear layer inference time was benchmarked on accelerator hardware across three forward execution strategies:

```
Strategy 1: Dense Baseline nn.Linear
  y = x W^T  (cuBLAS / GEMM)
  Latency: ~2.95 ms

Strategy 2: Multi-Core Einsum
  torch.einsum("bij, jqk, krl -> bil", x_reshaped, G1, G2)
  Latency: ~45.06 ms  (16.9x SLOWER)
  Bottleneck: Broken memory coalescing, 5D tensor indexing, zero Tensor Core usage.

Strategy 3: Pre-Contracted Tensordot + GEMM
  W_eff = torch.tensordot(G1, G2, dims=([3], [0]))  -> F.linear(x, W_eff)
  Latency: ~4.77 ms   (1.62x slower than baseline)
  Bottleneck: Intermediate tensor allocations; W is fully reconstructed in memory.
```

### Conclusions on Compute Efficiency:
- **No FLOPs reduction**: Because Strategy 3 materializes the full $W$ matrix prior to multiplication, total FLOPs are identical to or higher than dense GEMM.
- **Memory footprint**: Strategy 3 incurs additional activation memory during backward and forward passes to store intermediate contracted cores.
- **Strategy 2 is non-viable**: Direct multi-core contraction without materialization suffers severe GPU memory bandwidth starvation, resulting in a 17x slowdown.

---

## 5. Architectural Recommendations

1. **For 2D Linear Layers**:
   - Use **Truncated SVD (`svd_linear`, `asvd`, `aasvd`)** or **Adaptive LoRA**. SVD maintains optimal low-rank projection properties along canonical matrix axes, delivers 50%–75% parameter reduction, and directly reduces GEMM FLOPs by executing $y = (x V) U^T$.
2. **When TT is Truly Applicable**:
   - Embedding tables of extreme dimensionality ($V \times D$ where $V \ge 250,000$).
   - Multimodal tensor representations ($D \ge 4$ or 5 modes).
   - Higher-order convolutional kernels (e.g. 3D/4D spatio-temporal video convolutions).

---

## 6. Deprecation & Migration Guide

The TT modules remain available for archival and research verification under:
- Implementation: [`Codes/framework/compression_engine/tt_linear.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/tt_linear.py)
- Compressor: [`Codes/framework/compression_engine/tt_compressor.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/tt_compressor.py)
- Standalone Runners:
  - [`Codes/experimental/runners/run_all_4_tt.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/experimental/runners/run_all_4_tt.py)
  - [`Codes/experimental/runners/run_all_4_tt_linear.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/experimental/runners/run_all_4_tt_linear.py)

To evaluate TT in an isolated experimental session, run:
```bash
python Codes/experimental/runners/run_all_4_tt.py --models distilbert,roberta,gpt2
```
TT will **not** be selected by `run_all_benchmarks_queue.py`.

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.94%** | -0.12% | 0.9127 | 1.00x | 0 | [svd-linear-attention-distilbert-sst2-tt](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-tt) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **81.10%** | -0.70% | 0.8106 | 1.00x | 0 | N/A | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **94.70%** | +0.40% | 0.9432 | 1.00x | 0 | [svd-linear-attention-distilbert-agnews-full-tt](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-tt) | 2026-09-19 |