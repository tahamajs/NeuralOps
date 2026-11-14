# NeuralOps Interactive 3D Visualizer & Dashboard Suite

NeuralOps includes a high-performance WebGL / Three.js 3D architecture visualizer, an interactive Latent Manifold & SVD Spectral Decay engine, and an Ada Lovelace SM 8.9 Empirical Hardware Roofline analyzer. The suite enables real-time inspection, animation, and analytical auditing across 131 attention mechanisms and 72 model compression topologies.

---

## 1. Suite Overview & Architecture Components

The interactive dashboard suite comprises three core pillars:

1. **Interactive WebGL/Three.js Attention Atlas & Architecture Visualizer (`docs/visualizer/index.html`)**:
   - 3D spatial token routing graphs comparing quadratic ($\mathcal{O}(L^2)$) full attention, chunkwise recurrences, and linear ($\mathcal{O}(L)$) state propagation.
   - Dynamic 3D Bezier conduits mapping inter-module signal flows across Query/Key/Value projections, Multi-Query/Grouped-Query (GQA/MQA) sharing, SwiGLU bifurcations, and residual skip highways.
   - Level-of-Detail (LOD) rendering engine handling full deep transformer stacks (Macro overview down to tensor-level sub-operation drilldown).
   - Interactive head specialization visualizers and dynamic 2:4 structured sparsity pruning simulations.

2. **3D Latent Manifold & SVD Spectral Decay Visualizer (`docs/visualizer/latent_manifold.html`)**:
   - Continuous 3D parametric surface modeling singular value decay: $y(x, z) = e^{-\gamma \sqrt{x^2 + z^2}} \cos(0.25 \sqrt{x^2 + z^2}) \cdot 12 \cdot \frac{r}{64}$.
   - Real-time singular energy recovery calculation and rank truncation thresholding ($r \in [4, 64]$).
   - Dynamic spectral surface recoloring and mesh normal recomputation for projection factorizations (SVD, ASVD, HNC-SVD).

3. **Ada Lovelace SM 8.9 Roofline Analyzer (`docs/roofline_interactive.html` / `docs/visualization/roofline_interactive.html`)**:
   - Hardware-grounded Plotly empirical roofline model calibrated for NVIDIA GeForce RTX 4070 (AD104 Silicon, SM 8.9).
   - Dual-ceiling plotting: Theoretical peak (FP16 Boost @ 2475 MHz: 160.2 TFLOPS, 504.0 GB/s GDDR6 bandwidth, ridge point $I^* = 317.9\text{ FLOP/B}$) alongside measured sustained baseline (70.6 TFLOPS, 255.8 GB/s).
   - Interactive kernel placement across 131 attention mechanisms and 72 compression operators, with custom operational arithmetic intensity and FLOPS calculators.

---

## 2. Interactive WebGL/Three.js Attention Atlas

### 2.1 Spatial Token Routing & Receptive Field Projections
The Attention Atlas renders token particle trajectories moving through attention and compression blocks.
- **Attention Routing Families**:
  - **Quadratic Softmax Attention** ($\mathcal{O}(L^2)$): Dense all-to-all token mesh with causal triangular masking.
  - **Linear / Recurrent Attention** ($\mathcal{O}(L)$): Linear recurrent state passing with feature maps $\phi(Q)\phi(K)^T$ and recurrent hidden state updates.
  - **Chunkwise & Associative Scans**: Tiled intermediate chunk representations bridging prefill parallelism and decoding efficiency.
  - **SVD-Compressed Projections**: Token funneling visualizer illustrating rank dimension squeezing through bottleneck matrices $U_r \Sigma_r V_r^T$.

### 2.2 Color-Coded Signal Bus Architecture
The visualizer models transformer data paths with standardized 3D cubic Bezier conduits:

| Signal Type | Hex Code | Visual Role | Routing Semantics |
|---|---|---|---|
| **Input Stream** | `#10b981` | Emerald Green | Sequence input entering from embedding lookup |
| **Baseline Feed** | `#38bdf8` | Sky Blue | Standard full-rank forward tensor connections |
| **Compressed Feed** | `#2dd4bf` | Mint Cyan | Low-rank factorized / compressed forward paths |
| **GQA KV Bundle** | `#f59e0b` | Amber Orange | Shared Key/Value projections (e.g. Qwen, LLaMA-3) |
| **SwiGLU Branch** | `#a855f7` | Violet Purple | Bifurcated gate and up-projection cables |
| **Residual Attn** | `#0284c7` | Deep Azure | Attention identity skip highway ($+ X_{\text{in}}$) |
| **Residual FFN** | `#7c3aed` | Royal Indigo | Feed-forward identity skip highway ($+ X_{\text{mid}}$) |
| **Output Head** | `#f43f5e` | Crimson Rose | Unembedding projection to vocabulary logits |

### 2.3 Interactive Controls & Keybindings
- **Orbit Navigation**: Left-click + drag to rotate, right-click + drag to pan, scroll wheel to zoom.
- **Space**: Toggle particle / token flow animation play/pause.
- **Timeline Scrubber**: Scrub token flow through layer lifecycles (0 to 6).
- **Preset Quick Picker**: Select from 203 verified attention modules and compression engines in the catalog.
- **LOD Slider**: Dynamically switch between Macro (high-level model topology), Meso (block-level modules), and Micro (tensor ops and weight matrices).

---

## 3. 3D Latent Manifold & SVD Spectral Decay Visualizer

The latent manifold explorer (`latent_manifold.html`) provides a real-time 3D representation of latent representations and weight decomposition geometries.

### 3.1 Mathematical Foundation
When decomposing weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ via truncated SVD or whitening pullback $\tilde{W} = U_r \Sigma_r V_r^T$, the retained singular energy is given by:

$$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}$$

In the visualizer, the manifold elevation $y(x, z)$ simulates the energy density distribution across projection space:
- **Decay Sharpness ($\gamma$)**: Modulated by layer type (e.g., Early Attention $W_q$, Deep Attention $W_o$, MLP Gate, SVD Bottleneck).
- **Rank Cutoff ($r \in [4, 64]$)**: Interactive slider truncating non-dominant singular modes while tracking recovered energy percentage ($80.0\% - 99.5\%$).
- **Color Mapping**: Dynamic HSL vertex gradient shifting from high-energy singular modes (deep turquoise `#38bdf8`) to truncated residual tail dimensions.

### 3.2 Supported Layer Inspections
- `Layer 0 - Early Attention (W_q)`
- `Layer 6 - Mid Attention (W_k)`
- `Layer 12 - Deep Attention (W_o)`
- `Layer 18 - MLP Gate Projection`
- `Layer 23 - SVD Bottleneck Layer`

---

## 4. Ada Lovelace SM 8.9 Hardware Roofline Analyzer

The interactive roofline visualizer (`roofline_interactive.html`) maps operational intensity against physical GPU hardware limits.

### 4.1 NVIDIA GeForce RTX 4070 (AD104 Silicon) Parameters
- **Architecture**: SM 8.9 (Ada Lovelace)
- **FP16 Tensor Core Peak (Boost @ 2475 MHz)**: $160.2\text{ TFLOPS}$
- **FP16 Tensor Core Peak (Base @ 2175 MHz)**: $141.6\text{ TFLOPS}$
- **FP8 Tensor Core MMA Peak**: $320.4\text{ TFLOPS}$
- **FP32 Vector ALU Peak**: $29.15\text{ TFLOPS}$
- **Memory Subsystem**: $12\text{ GB GDDR6}$, $192\text{-bit}$ bus width
- **Theoretical Peak Bandwidth**: $504.0\text{ GB/s}$
- **Measured Sustained Ceiling**: $70.6\text{ TFLOPS}$, $255.8\text{ GB/s}$
- **Hardware Ridge Point ($I^*$)**:

$$I^* = \frac{\text{Peak Throughput}}{\text{Peak Bandwidth}} = \frac{160.2 \times 10^{12}\text{ FLOP/s}}{504.0 \times 10^9\text{ Bytes/s}} \approx 317.9\text{ FLOP/byte}$$

### 4.2 Interactive Plotly Capabilities
- **Operational Intensity Regions**:
  - **Memory-Bound ($I < I^*$)**: Kernels limited by GDDR6 DRAM bandwidth ($P = I \times \text{Bandwidth}$).
  - **Compute-Bound ($I \ge I^*$)**: Kernels saturating SM 8.9 4th-generation Tensor Cores ($P = P_{\text{max}}$).
- **Kernel Datasets**: Real telemetry points for all 131 attention mechanisms and 72 compression engines.
- **Custom Kernel Calculator**: Input operation counts (MFLOPs / GFLOPs) and DRAM transfer bytes to compute exact arithmetic intensity, plot attained performance, and check distance to the theoretical hardware ceiling.

---

## 5. Deployment and Local Usage

### 5.1 Interactive Cloud Playground
The 3D neural architecture visualizer is deployed live on Hugging Face Spaces:
- **Live Space:** [https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer)

### 5.2 Local Execution via HTTP Server
Serve the entire visualization suite locally with zero npm or external dependencies:

```bash
# From repository root:
python3 -m http.server 8000 --directory docs/visualizer
```

Access points:
- **Attention Atlas & Architecture Visualizer**: `http://localhost:8000/index.html`
- **Latent Manifold Explorer**: `http://localhost:8000/latent_manifold.html`
- **Self-Contained Offline Visualizer**: `open docs/visualizer/standalone.html`

To serve the root visualization directory (including the Roofline analyzer):
```bash
python3 -m http.server 8000 --directory docs
```
- **Roofline Analyzer**: `http://localhost:8000/roofline_interactive.html`

### 5.3 NeuralOps CLI Launch
NeuralOps CLI provides direct launch shortcuts:

```bash
# Launch visualizer in default browser
neuralops visualizer --port 8000
```
