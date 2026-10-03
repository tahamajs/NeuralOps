# Attention Lab / Transformer Visualizer: Detail Audit Report

> **Generated At**: 2026-09-19  
> **Repository Target**: `Hooshaai/3d-neural-architecture-visualizer`  
> **Workspace**: `Docs/visualizer`

---

## 1. Inspector Audit (Right Panel)

| Section | Status | Source Data File | Missing Fields / Gaps | Verification & Details |
| :--- | :---: | :--- | :--- | :--- |
| **Structural Transformation** | ⚠️ Incomplete | `Docs/visualizer/src/educational/transformation.js` | Live per-module latency deltas and FLOP measurements are computed via analytical heuristics rather than precomputed JSON tables. | Baseline and compressed parameters & memory storage are verified from `modules-catalog.json`. Live inference benchmark latencies exist in `master_benchmark_summary.json` and need dynamic linking. |
| **Token Lifecycle & Dataflow** | ✅ Complete | `Docs/visualizer/src/educational/token_lifecycle.js` | None | All 7 stages ($X_{\text{in}} \to \text{LN} \to \text{Attn} \to + \to \text{FFN} \to + \to X_{\text{out}}$) are rendered with KaTeX math, dimension badges, and camera focus links. |
| **Architectural Component Pipeline** | ✅ Complete | `Docs/visualizer/src/educational/component_pipeline.js` | None | Hierarchical tree renders individual sublayers, tensor shapes, and parameters parsed from `modules_components.json`. |
| **Mathematical Formulation** | ✅ Complete | `Docs/visualizer/src/educational/formula_renderer.js` | None | Renders canonical LaTeX with KaTeX and supports instant clipboard copying. |
| **Module Description & Implementation** | ✅ Complete | `Docs/visualizer/src/ui/inspector.js` | None | Displays verified source file path, PyTorch classes, description, and opens full research paper modal. |
| **4-Model Architecture Matrix** | ✅ Complete | `Docs/visualizer/src/educational/models_matrix.js` | None | Displays side-by-side DistilBERT, RoBERTa, GPT-2, and Qwen benchmarks with SVG sparklines. |

---

## 2. 3D Scene Audit

| Checkpoint | Status | Verified Details |
| :--- | :---: | :--- |
| **Module Name Readability** | ✅ Complete | Rendered via 3D canvas billboard sprites ($1024 \times 256$ canvas resolution) with high contrast. No text truncation. |
| **Semantic Colors** | ✅ Complete | SVD Low-Rank = Teal (`#2DD4BF`), MoE = Purple (`#A855F7`), MLA = Amber (`#F59E0B`), Dense Baseline = Blue (`#3B82F6`), Residual Highway = Emerald Green (`#10B981`). |
| **Real Tensor Dimensions** | ✅ Complete | Displayed on floating 3D badges in near zoom (e.g. `[768 × 768]`, `[768 × 64]`, `[3072 × 768]`). |
| **Status Indication** | ✅ Complete | Dual-column comparative layout: Baseline column at $x = -4.5$, Compressed column at $x = +4.5$. |
| **Token Physics Meaning** | ✅ Complete | Particles follow exact trajectories from `computeToken3DTrajectory`: 75% internal compute path (Q/K/V, SVD neck, MoE routes), 25% lateral residual skip highway path ($x = \text{worldX} \pm 1.85$) merging into $(+)$ addition nodes. |
| **Center Space Check** | ✅ Complete | No decorative white box; open spatial viewing corridor with free orbit controls. |

---

## 3. Mathematical Formula Audit

| Attention / Module Type | Rendered KaTeX Formula | Verification against Python Source | Source File | Status |
| :--- | :--- | :--- | :--- | :---: |
| **MHA** | $\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O$ | Matches standard multi-head projection in PyTorch `nn.MultiheadAttention` | `Codes/framework/attentions/standard.py` | ✅ Verified |
| **MQA** | $\text{MQA}(Q, K_1, V_1) = \text{Concat}(\text{Softmax}(Q_h K_1^\top / \sqrt{d_k}) V_1) W_O$ | Verified shared single-head Key and Value tensors | `Docs/visualizer/src/educational/attention_types.js` | ✅ Verified |
| **GQA** | $\text{GQA}(Q_g, K_g, V_g) = \text{Concat}(\text{Attn}(Q_{g,h}, K_g, V_g)) W_O$ | Matches Qwen / Llama-3 grouped query repetition mechanism | `Codes/framework/attentions/gqa.py` | ✅ Verified |
| **MLA** | $c_t^{KV} = W^{DKV} x_t, \quad K_t^C = W^{UK} c_t^{KV}, \; V_t^C = W^{UV} c_t^{KV}$ | Matches DeepSeek-V2/V3 latent down-projection and uncompressed RoPE decoupling | `Codes/framework/attentions/deepseek_mla.py` | ✅ Verified |
| **Fused QKV** | $[Q, K, V] = \text{Split}_3(X W_{\text{c\_attn}}), \quad W_{\text{c\_attn}} \in \mathbb{R}^{d \times 3d}$ | Matches Hugging Face GPT-2 `Conv1D` fused projection operator | `Codes/framework/attentions/fused_qkv.py` | ✅ Verified |
| **SVD Attention** | $W \approx U_r \Sigma_r V_r^\top = A B^\top, \quad r \ll d_{\text{model}}$ | Matches truncated SVD linear factorization $V_{768 \to 64} \times U_{64 \to 768}$ | `Codes/framework/attentions/svd_linear.py` | ✅ Verified |
| **Linear Attention** | $\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \sum_j \phi(K_j)^\top}$ | Matches non-negative kernel feature map associative formulation | `Codes/framework/attentions/linear_attention.py` | ✅ Verified |

---

## 4. Attention Module Audit (11 Architectures)

| Variant | In 3D Scene? | Inspector Explanation? | Formula? | KV Cache Footprint | Benchmarks? | Source File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MHA** | ✅ Yes | ✅ Yes | ✅ Yes | $2 \times 12 \times 64 \times 2 = 3,072\text{ B/tok}$ | ✅ 92.4% SST-2 | `Codes/framework/attentions/standard.py` |
| **MQA** | ✅ Yes | ✅ Yes | ✅ Yes | $2 \times 1 \times 64 \times 2 = 256\text{ B/tok}$ | ⚠️ Theoretical | `Docs/visualizer/src/educational/attention_types.js` |
| **GQA** | ✅ Yes | ✅ Yes | ✅ Yes | $2 \times 2 \times 64 \times 2 = 512\text{ B/tok}$ | ✅ 93.5% SST-2 | `Codes/framework/attentions/gqa.py` |
| **MLA** | ✅ Yes | ✅ Yes | ✅ Yes | $512 \times 2 = 1,024\text{ B/tok}$ | ⚠️ Theoretical | `Codes/framework/attentions/deepseek_mla.py` |
| **Fused QKV** | ✅ Yes | ✅ Yes | ✅ Yes | $2 \times 12 \times 64 \times 2 = 3,072\text{ B/tok}$ | ✅ 90.2% SST-2 | `Codes/framework/attentions/fused_qkv.py` |
| **Sliding Window** | ✅ Yes | ✅ Yes | ✅ Yes | $O(W)$ bounded cache | ⚠️ Theoretical | `Docs/visualizer/src/educational/attention_types.js` |
| **Sparse Attention** | ✅ Yes | ✅ Yes | ✅ Yes | Block-sparse gather cache | ⚠️ Theoretical | `Docs/visualizer/src/educational/attention_types.js` |
| **Linear Attention** | ✅ Yes | ✅ Yes | ✅ Yes | Fixed state $O(d^2)$ cache | ✅ Verified | `Codes/framework/attentions/linear_attention.py` |
| **Flash Attention** | ✅ Yes | ✅ Yes | ✅ Yes | Standard MHA cache (exact SRAM tiling) | ✅ Verified | `Docs/visualizer/src/educational/attention_types.js` |
| **Cross-Attention** | ✅ Yes | ✅ Yes | ✅ Yes | Fixed encoder cache | ⚠️ Theoretical | `Docs/visualizer/src/educational/attention_types.js` |
| **Causal Attention** | ✅ Yes | ✅ Yes | ✅ Yes | Autoregressive lower-tri cache | ✅ 90.2% SST-2 | `Docs/visualizer/src/educational/attention_types.js` |

---

## 5. Summary & Action Plan for Priority Tier 1

1. **Feature 1.1**: Build `Docs/visualizer/src/interactive/attention_heatmap.js` allowing users to type any sentence and inspect full `[layers, heads, seq, seq]` attention affinity matrices in interactive SVG/Canvas with row/column highlights, hover tooltips, and PNG export.
2. **Feature 1.2**: Implement Live Token Prediction Playground (`token_prediction.js`) with top-10 probability bars and temperature / top-k sliders.
3. **Feature 1.3**: Implement Compression Simulator (`compression_simulator.js`) with real-time rank slider and Pareto trade-off curve.
4. **Feature 1.4**: Implement Side-by-Side Model Race (`model_race.js`) comparing latency, peak memory, and FLOPs.
5. **Feature 1.5**: Implement Shareable Permalink per Module with URL query parameters.
6. **Feature 1.6**: Implement Video Exporter (`video_exporter.js`) recording automated 3D tour via MediaRecorder.
7. **Feature 1.7**: Implement Guided Lessons (`src/educational/lessons/`) with interactive exercises and quizzes.
