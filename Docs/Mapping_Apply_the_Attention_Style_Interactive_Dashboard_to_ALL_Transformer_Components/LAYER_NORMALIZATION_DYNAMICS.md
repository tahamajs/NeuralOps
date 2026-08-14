# 🧪 LAYER NORMALIZATION DYNAMICS — "The Stabilizer Lab"

## Complete Implementation Blueprint (No Code — Full Architecture & Technology Stack)

---

## 📐 PART 0: ARCHITECTURE PHILOSOPHY — The Four-Tier Computation Stack

The Stabilizer Lab requires a **hybrid computation architecture** because it must simultaneously: (a) run real ML model inference to extract hidden states before/after normalization, (b) perform heavy mathematical computations (mean, variance, RMS, gradient norms), (c) render GPU-accelerated 3D vector field visualizations, and (d) keep the UI responsive with 60fps interactivity. No single technology can handle all four demands alone, so we use a **four-tier stack**:

| Tier | Technology | Responsibility | Why This Choice |
|------|-----------|----------------|-----------------|
| **Tier 1 — Browser ML Inference** | Transformers.js (Xenova) + ONNX Runtime Web with WebGPU backend | Run actual GPT-2/BERT models in-browser to extract hidden states before/after LayerNorm. No server needed. | Transformers.js "uses ONNX Runtime to run models in the browser" and "WebGPU + JavaScript: How Browsers Became Real AI Runtimes" proves this is production-ready. Firefox's native ONNX Runtime integration demonstrates "accelerating inference by replacing the default onnxruntime-web with its native C++ counterpart." |
| **Tier 2 — Scientific Computation** | Pyodide (Python-in-Wasm) with NumPy + SciPy | Compute per-layer statistics (mean, variance, RMS, gradient norms), run PCA/t-SNE for vector field projections, compute cosine similarity between pre/post normalization vectors. Runs in a Web Worker to avoid blocking the UI. | Pyodide "includes NumPy, Pandas, Matplotlib, and SciPy out-of-the-box" enabling "powerful data analysis, math, and visualization right inside the browser." The Web Worker with SharedArrayBuffer approach enables "zero-copy data transfer" between the ML thread and the computation thread. |
| **Tier 3 — GPU-Accelerated Visualization** | React Three Fiber (@react-three/fiber + @react-three/drei) + Deck.gl + ECharts-GL | Render 3D vector field (how LayerNorm maps vectors onto unit sphere), 3D bar charts for γ/β parameters, real-time line charts for norm statistics, heatmaps for gradient flow. | Helios-Web demonstrates "real-time visualization and interactive analysis of massive-scale networks and high-dimensional embeddings" using "GPU-based rendering, modern web technologies (WebGL and WebAssembly)." ECharts-GL provides "3D visualization capabilities through WebGL" with "hardware acceleration for large-scale datasets." Deck.gl is "GPU-driven" and handles "millions of data points." |
| **Tier 4 — Reactive UI & Dashboard** | React 18 + TypeScript + Tailwind CSS + Framer Motion + Zustand | Dashboard layout, zone management, user interactions, state synchronization across zones, animation orchestration. | Standard modern stack; Framer Motion for layout animations; Zustand for lightweight global state without Redux boilerplate. |

### Architecture Diagram (Data Flow)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE STABILIZER LAB                            │
│                                                                      │
│  USER INPUT: Text + Model Selection                                  │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ TIER 1: ML INFERENCE ENGINE (Web Worker #1)                  │   │
│  │ ┌────────────────────┐  ┌────────────────────────────────┐   │   │
│  │ │ Transformers.js    │  │ ONNX Runtime Web (WebGPU)      │   │   │
│  │ │ (Xenova)           │  │ - Run GPT-2 / BERT             │   │   │
│  │ │ - Load model       │  │ - Extract hidden states per    │   │   │
│  │ │ - Tokenize text    │  │   layer BEFORE LayerNorm       │   │   │
│  │ │ - Run forward pass │  │ - Extract hidden states AFTER  │   │   │
│  │ │                    │  │   LayerNorm                    │   │   │
│  │ └────────────────────┘  └────────────────────────────────┘   │   │
│  │                         │                                     │   │
│  │         SharedArrayBuffer (zero-copy)                         │   │
│  │                         ▼                                     │   │
│  │ ┌────────────────────────────────────────────────────────┐   │   │
│  │ │ TIER 2: SCIENTIFIC COMPUTE (Web Worker #2 — Pyodide)    │   │   │
│  │ │ - NumPy: mean, var, RMS per layer                       │   │   │
│  │ │ - SciPy: PCA of pre/post-norm vectors                   │   │   │
│  │ │ - Compute cosine similarity (pre vs post)               │   │   │
│  │ │ - Compute gradient norms per layer                      │   │   │
│  │ │ - Compute vector field transformation (sphere mapping)  │   │   │
│  │ └────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼ (Computed data → Main Thread via postMessage)               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ TIER 3: GPU VISUALIZATION (Main Thread — WebGL/WebGPU)       │   │
│  │ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │   │
│  │ │ React Three  │ │ Deck.gl      │ │ ECharts-GL           │  │   │
│  │ │ Fiber        │ │ - Heatmaps   │ │ - 3D Bar Charts      │  │   │
│  │ │ - 3D Vector  │ │ - Line charts│ │ - Line Charts        │  │   │
│  │ │   Field      │ │ - Scatter    │ │ - Surface Plots      │  │   │
│  │ │ - Unit Sphere│ │   plots      │ │                      │  │   │
│  │ │ - Particle   │ │              │ │                      │  │   │
│  │ │   Systems    │ │              │ │                      │  │   │
│  │ └──────────────┘ └──────────────┘ └──────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ TIER 4: REACTIVE UI (React 18 + Tailwind + Framer Motion)    │   │
│  │ - Dashboard shell with 5 zones                               │   │
│  │ - Zustand global state                                       │   │
│  │ - React Query for data fetching                              │   │
│  │ - Framer Motion for layout animations                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ PART 1: ZONE-BY-ZONE IMPLEMENTATION DETAILS

---

### ZONE A: NORM STATISTICS DASHBOARD — "The Vital Signs Monitor"

**What it shows:** Per-layer, a comprehensive dashboard of what happens to hidden states before and after LayerNorm. Think of it as the "vital signs monitor" for each transformer layer — you can instantly see if a layer is healthy (stable activations) or sick (vanishing/exploding activations).

#### Sub-Feature A1: Pre/Post Mean & Variance Line Charts

**Implementation:**

- **Data Source:** Pyodide Web Worker computes `np.mean(hidden_states, axis=-1)` and `np.var(hidden_states, axis=-1)` for each layer, both before and after LayerNorm, across all tokens in the input sequence.
- **Visualization Library:** **ECharts-GL** (not D3 or Plotly). Why? Because with 12+ layers × 768+ dimensions × potentially hundreds of tokens, we need GPU-accelerated rendering for smooth pan/zoom. ECharts-GL "extends the core ECharts library to enable the creation of 3D charts... while maintaining a similar API to standard ECharts." The `renderer: 'webgl'` option activates hardware acceleration.
- **Rendering Strategy:** Two overlaid line series per chart: (1) Pre-LayerNorm values in **orange/red** (indicating "before stabilization"), (2) Post-LayerNorm values in **blue/green** (indicating "after stabilization"). The Y-axis shows the statistic value; the X-axis shows layer index.
- **Interactivity:** Hover over any point to see the exact numerical value. Click a layer to "zoom into" that layer and see per-dimension breakdown in Zone B. A toggle switches between "Mean View" and "Variance View."
- **Performance:** For 12 layers × 768 dimensions, ECharts-GL with WebGL backend renders 60fps even with animated transitions. The "Performance Optimization" page documents "techniques and best practices for optimizing the performance of 3D visualizations in ECharts-GL."

#### Sub-Feature A2: Distribution Histogram (Per-Layer Pre/Post)

**Implementation:**

- **Data Source:** Pyodide computes histograms of hidden state values for each layer, before and after LayerNorm.
- **Visualization:** **Deck.gl** `HexagonLayer` or standard ECharts histogram. Deck.gl is chosen because it's "GPU-driven" and can handle millions of data points — important when we're binning 768 dimensions × 128 tokens = ~98K values per layer.
- **Rendering:** Two overlapping semi-transparent histograms: pre-norm in red, post-norm in blue. The post-norm distribution should be visibly more "compact" (around mean 0, variance 1). The before-norm distribution reveals if the layer has activation drift or exploding values.
- **Animation:** When the user changes the selected layer, the histogram smoothly transitions between layers using Deck.gl's built-in transition system.

#### Sub-Feature A3: Gradient Flow Through Normalization

**Implementation:**

- **Data Source:** This requires running a backward pass through the model. Transformers.js supports this? Actually, for a production system, we use the **ONNX Runtime Web with WebGPU** to compute gradients via a custom training loop. Alternatively, we pre-compute gradients server-side and stream them via WebSocket (using your existing Channels infrastructure).
- **Visualization:** A heatmap (Deck.gl `ScreenGridLayer` or ECharts heatmap) showing gradient magnitude per layer × per training step (or per input token). The Visual Sandbox paper "gains insights into training dynamics, circuit transferability, and the causes of loss spikes, including those induced by the high curvature of normalization layers" — this is exactly what we want to visualize.
- **Color Scale:** Green = healthy gradient flow, Yellow = warning (gradient getting small), Red = critical (vanishing gradient). The Pre-LN vs Post-LN comparison from the literature "clearly visualizes the instability and spikes inherent in Post-LN, validating the modern preference for Pre-LN stability."
- **Key Insight Display:** A prominent banner shows: "Layer 7 gradient norm: 0.003 (⚠️ Vanishing)" or "Layer 3 gradient norm: 1.2 (✅ Healthy)" based on configurable thresholds.

---

### ZONE B: VECTOR FIELD VISUALIZATION — "The Sphere Mapper"

**What it shows:** A 3D interactive view of how LayerNorm transforms vectors. Before LayerNorm, vectors are scattered throughout space. After LayerNorm, they're projected onto (or near) a unit sphere. This is the most visually striking and educationally powerful feature.

#### Sub-Feature B1: The Unit Sphere + Pre/Post Vectors

**Implementation:**

- **Visualization Library:** **React Three Fiber (@react-three/fiber + @react-three/drei)** — this is the gold standard for 3D in React. TensorSpace.js "translates neural network data into interactive 3D objects using THREE.js" and "enables the 3D representation of neural network components, handling both the visual elements and the interaction with neural network data."
- **Scene Composition:**
  1. **Unit Sphere:** A semi-transparent wireframe sphere (radius=1) rendered with `<Sphere>` from @react-three/drei. The sphere represents the "target" — after LayerNorm, all vectors should lie on or near this sphere.
  2. **Pre-Norm Vectors:** Each token's hidden state (before LayerNorm) is rendered as an **arrow** pointing from the origin to its vector position. Color-coded by token index (using a viridis colormap). These arrows will be scattered at various distances from the origin.
  3. **Post-Norm Vectors:** Each token's hidden state (after LayerNorm) is rendered as a **dot on (or near) the sphere surface**. The same color coding shows which pre-norm vector mapped to which post-norm position.
  4. **Connection Lines:** Dashed lines connect each token's pre-norm position to its post-norm position, showing the "journey" through LayerNorm.
- **Camera Controls:** `<OrbitControls>` from @react-three/drei allows rotation, zoom, and pan. The initial camera position shows the sphere from a slight angle so the 3D nature is immediately apparent.
- **Animation:** When the user changes the layer or input text, the vectors **animate** from their old positions to new positions using `react-spring/three` or Framer Motion 3D. This is crucial — without animation, users can't track which vector moved where.
- **Performance:** For 128 tokens × 768 dimensions, we use PCA (computed in Pyodide) to reduce to 3 dimensions before rendering. The 3D scene handles 128 arrows + 128 dots + 128 lines = ~384 objects, which React Three Fiber renders effortlessly at 60fps.

#### Sub-Feature B2: LayerNorm "Magnifying Glass" — Single Vector Detail

**Implementation:**

- When the user **clicks on a single vector** (either pre-norm arrow or post-norm dot), a "magnifying glass" panel appears showing:
  - The **raw 768-dimensional vector** as a mini bar chart (too many bars to be readable, but gives a sense of distribution)
  - The **mean and variance** of that specific token
  - The **cosine similarity** between pre-norm and post-norm vectors (computed in Pyodide)
  - The **LayerNorm formula** applied to this specific vector, showing: `x - μ = [...]`, `σ = [...]`, `(x - μ)/σ = [...]`, `γ ⊙ (x-μ)/σ + β = [...]`
  - This is the "3Blue1Brown moment" — the user sees the math applied to concrete numbers, making the abstract formula tangible.
- **Implementation Approach:** The formula rendering uses **KaTeX** (fastest LaTeX renderer for web) with live-substituted numerical values. The Transformer Explainer similarly "visualizes different operations within the transformer block: Layer Normalization — visualized as a curved line and a circle with tooltips."

---

### ZONE C: γ/β PARAMETER VISUALIZATION — "The Scaling/Shifting Dashboard"

**What it shows:** LayerNorm has two learnable parameter vectors: **γ (scale)** and **β (shift)**, each of dimension `d_model`. This zone shows which dimensions the model has learned to amplify vs. suppress, revealing the model's "feature importance" per layer.

#### Sub-Feature C1: 3D Bar Chart of γ and β

**Implementation:**

- **Visualization Library:** **ECharts-GL** `Bar3D` chart type. ECharts-GL "provides 3D visualization capabilities through WebGL" including "3D bar charts." The 3D nature is important because 768 dimensions cannot be shown in a 2D bar chart without scrolling.
- **Rendering:** The X-axis represents dimension index (0 to 767), the Z-axis represents the parameter value, and the Y-axis can represent different layers (or vice versa). Bars for γ are colored **warm** (red/orange) for values > 1 (amplifying), **cool** (blue) for values < 1 (suppressing). Bars for β are colored **green** for positive shifts, **purple** for negative shifts.
- **Interactivity:** The user can rotate the 3D view, zoom into specific dimension ranges, and hover over any bar to see exact values. A "Top 10" panel lists the most amplified and most suppressed dimensions.
- **Alternative View:** A 2D heatmap (using Deck.gl or ECharts) where rows = layers, columns = dimension ranges (binned into groups of 32), and color = average γ or β value.

#### Sub-Feature C2: γ/β Distribution Histogram

- A simple histogram showing the distribution of γ values and β values across all dimensions. Are they clustered around 1 and 0 (the initialization values)? Has the model learned to deviate significantly?
- **Library:** ECharts with `renderer: 'webgl'` for smooth rendering.

---

### ZONE D: MATH PANEL — "The Formula Lab"

**What it shows:** Interactive, step-by-step visualization of the LayerNorm and RMSNorm formulas, with live numerical values from the currently selected token/vector.

#### Sub-Feature D1: LayerNorm Step-by-Step Animator

**Implementation:**

- **Math Rendering:** **KaTeX** (via `react-katex`) for LaTeX formula rendering. KaTeX is "the fastest math typesetting library for the web."
- **Animation Engine:** **Framer Motion** for step-by-step reveal animations. The formula is displayed as a sequence of steps:
  1. `x = [0.847, -0.312, 1.203, ...]` — show the raw vector
  2. `μ = 0.042` — compute and display the mean
  3. `σ² = 0.891` — compute and display the variance
  4. `(x - μ) / √(σ² + ε) = [...]` — show the normalized vector
  5. `γ ⊙ norm + β = [...]` — show the final output
- **Each step** reveals with a Framer Motion animation (fade + slide). The numerical values update in real-time as the user changes the selected token (by clicking on a vector in the 3D view) or the selected layer.
- **Color coding:** Raw values in gray, computed statistics in yellow, normalized values in green, final output in blue.

#### Sub-Feature D2: RMSNorm Comparison Mode

**Implementation:**

- **Side-by-side display** of LayerNorm and RMSNorm formulas. The key difference (RMSNorm skips mean subtraction) is **highlighted in red**.
- **Numerical comparison table:** For the same input vector, show:
  - LayerNorm output vs. RMSNorm output
  - Cosine similarity between the two outputs
  - Computational cost comparison (RMSNorm "少了求均值和减均值两个操作，在大模型里能明显提速")
- **Performance note:** A small badge shows: "RMSNorm: 2 fewer operations per dimension → ~15% faster in large models."

#### Sub-Feature D3: Interactive Parameter Tweaking

**Implementation:**

- Two **sliders** (using your existing `Slider` component) allow the user to manually adjust γ and β values for any dimension.
- As the user drags the slider, the 3D vector field (Zone B) updates in real-time, showing how the vector moves.
- This is the "what if" experimentation mode — researchers can test hypotheses like "What if γ was 2.0 for dimension 42?"

---

### ZONE E: INTERPRETABILITY & DIAGNOSTICS — "The Doctor's Toolkit"

#### Sub-Feature E1: Norm Ablation — What Happens If We Remove LayerNorm?

**Implementation:**

- **Data Source:** This requires running the model twice — once normally, once with LayerNorm bypassed (identity function). Pyodide can't do this efficiently; we use the **Django backend** to run the ablation and stream results via WebSocket (your existing Channels infrastructure).
- **Visualization:** A side-by-side comparison:
  - **Left:** Attention patterns WITH LayerNorm
  - **Right:** Attention patterns WITHOUT LayerNorm
  - **Bottom:** A "difference heatmap" showing where attention changed most
- **Metrics display:** "Without LayerNorm at Layer 5: attention entropy increased by 23%, output probability of correct token dropped from 0.87 to 0.12."

#### Sub-Feature E2: Pre-LN vs. Post-LN Architecture Comparison

**Implementation:**

- **Diagram Rendering:** Use **D3.js** (not a charting library — raw D3 for custom SVG) to draw two architecture diagrams side-by-side:
  - **Post-LN (original Transformer):** `x → Attention → Add → LayerNorm → FFN → Add → LayerNorm`
  - **Pre-LN (modern):** `x → LayerNorm → Attention → Add → LayerNorm → FFN → Add`
- **Gradient Flow Comparison:** Below each diagram, show a line chart of gradient norms per layer during training. Post-LN typically shows a U-shaped curve (large gradients at first and last layers, small in middle), while Pre-LN shows a flatter, more uniform distribution. The literature "clearly visualizes the instability and spikes inherent in Post-LN, validating the modern preference for Pre-LN stability."
- **Animation:** The user can scrub through "training steps" to see how gradient flow evolves over time.

#### Sub-Feature E3: RMSNorm vs. LayerNorm Side-by-Side Analyzer

**Implementation:**

- **Data Source:** The Django backend runs the same model twice — once with LayerNorm, once with RMSNorm substituted — and streams the hidden states for comparison.
- **Visualization:** Three panels:
  1. **Statistics comparison table:** Per layer, show mean, variance, and RMS for both normalization methods.
  2. **Output cosine similarity:** Per layer, how similar are the outputs of LayerNorm vs. RMSNorm? (Should be very high for most layers.)
  3. **Performance comparison:** Bar chart showing FLOPs per layer for each method.
- **Key Insight:** RMSNorm "保持非零均值分布，不执行均值对零的对齐操作" — the visualization should make this geometric difference visible.

---

## 🔧 PART 2: FRAMEWORK-BY-FRAMEWORK SETUP GUIDE

### 2.1 Transformers.js (Browser ML Inference)

```bash
npm install @xenova/transformers
```

**Key Configuration:**
- Use `pipeline('feature-extraction', model_name)` to get hidden states.
- For attention-specific work, use the lower-level API to extract per-layer hidden states before/after LayerNorm.
- Enable `quantized: true` for faster loading (the model runs in-browser, so quantization is essential).
- The library "uses ONNX Runtime to run models in the browser" and supports WebGPU backend for hardware acceleration.

### 2.2 Pyodide (Scientific Computing in Browser)

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
```

**Key Configuration:**
- Load in a **Web Worker** (not the main thread) to avoid blocking the UI.
- Pre-load packages: `await pyodide.loadPackage(['numpy', 'scipy'])`.
- Use `SharedArrayBuffer` for zero-copy data transfer between the ML worker and the compute worker.
- Pyodide "includes NumPy, Pandas, Matplotlib, and SciPy out-of-the-box" enabling "powerful data analysis, math, and visualization right inside the browser."

### 2.3 React Three Fiber (3D Visualization)

```bash
npm install @react-three/fiber @react-three/drei three
```

**Key Configuration:**
- Use `<Canvas>` with `gl={{ antialias: true, alpha: true }}` for high-quality rendering.
- Use `<OrbitControls>` for camera interaction.
- Use `react-spring/three` for smooth animation transitions.
- Reference implementation: TensorSpace.js "translates neural network data into interactive 3D objects using THREE.js."

### 2.4 Deck.gl (GPU-Accelerated 2D/3D Charts)

```bash
npm install deck.gl @deck.gl/react @deck.gl/layers
```

**Key Configuration:**
- Use `DeckGL` component with `WebGLRenderer`.
- Primary layers: `ScatterplotLayer` (vector field points), `LineLayer` (connections), `HeatmapLayer` (gradient flow).
- Deck.gl is "GPU-driven" and handles "millions of data points."

### 2.5 ECharts-GL (3D Bar Charts, Line Charts)

```bash
npm install echarts echarts-gl
```

**Key Configuration:**
- Register ECharts-GL: `import 'echarts-gl'`.
- Set `renderer: 'webgl'` in chart options for GPU acceleration.
- Primary chart types: `bar3D`, `line3D`, `scatter3D`, `surface`.
- ECharts-GL "extends the core ECharts library to enable the creation of 3D charts, globe visualizations, and other WebGL-accelerated graphics."

### 2.6 KaTeX (Formula Rendering)

```bash
npm install katex react-katex
```

**Key Configuration:**
- Use `\\displaystyle` for full-size formulas.
- Use `\\color{...}` for color-coded formula elements.
- Wrap in `React.memo` since formulas don't change often.

---

## 🔄 PART 3: DATA FLOW ARCHITECTURE

### 3.1 Initial Load Sequence

```
1. User enters text + selects model
2. Transformers.js loads model in Web Worker #1 (may take 5-15 seconds for first load)
3. Model runs forward pass → hidden states extracted at each layer (before/after LN)
4. Hidden states transferred to Web Worker #2 via SharedArrayBuffer
5. Pyodide computes statistics (mean, var, RMS, PCA)
6. Computed data posted back to main thread
7. All 5 zones render simultaneously with smooth entrance animations
```

### 3.2 Real-Time Update Sequence (User Changes Input)

```
1. User changes text or layer selection
2. If text changed: Steps 3-6 from above repeat
3. If only layer changed: Only Zone B (3D view) and Zone C (γ/β) need updates
4. Zone A (statistics) updates the highlighted layer marker
5. All transitions animated with Framer Motion (200ms duration)
```

### 3.3 WebSocket Stream (For Training Dynamics / Gradient Flow)

```
1. Django backend runs training loop
2. At each logging step, backend sends WebSocket message:
   { layer: 5, pre_mean: 0.04, post_mean: 0.00, gradient_norm: 0.023, ... }
3. Frontend Zustand store updates
4. Zone A (statistics line charts) appends new data point
5. Zone D (math panel) shows live-updating gradient values
```

---

## 📊 PART 4: PERFORMANCE STRATEGY

### 4.1 What Runs Where

| Computation | Location | Justification |
|-------------|----------|---------------|
| Model inference (forward pass) | Web Worker #1 (Transformers.js + ONNX Runtime Web) | Avoids blocking main thread; uses WebGPU for GPU acceleration |
| Statistical computation (NumPy) | Web Worker #2 (Pyodide) | Heavy computation offloaded; SharedArrayBuffer enables zero-copy data transfer |
| 3D rendering (Three.js) | Main thread (WebGL) | Must be on main thread for DOM access; GPU-accelerated |
| 2D charts (ECharts-GL, Deck.gl) | Main thread (WebGL) | GPU-accelerated; lightweight compared to 3D |
| React UI updates | Main thread | Standard React rendering; Zustand for efficient state updates |

### 4.2 Memory Budget

- **Model weights (GPT-2 small):** ~500MB in browser memory (quantized to 8-bit)
- **Hidden states cache (12 layers × 2 states × 768 dims × 128 tokens × 4 bytes):** ~9.4MB
- **3D scene geometry (vectors, sphere, lines):** ~2MB GPU memory
- **Total browser memory:** <600MB — feasible for modern browsers

### 4.3 Fallback Strategy

- If WebGPU is unavailable → fallback to WebAssembly (ONNX Runtime Web WASM backend)
- If Pyodide fails to load → fallback to pure JavaScript computation (using `math.js` or manual implementation)
- If WebGL is unavailable → degrade 3D view to 2D projection using ECharts with Canvas renderer

---

## 📋 PART 5: ZUSTAND STORE DESIGN (Global State)

```
Store: useStabilizerLabStore
├── modelId: string                    // which model is loaded
├── inputText: string                  // user's input text
├── selectedLayer: number              // which layer is selected
├── selectedTokenIndex: number | null  // which token is selected (clicked in 3D)
├── selectedDimRange: [number, number] // which dimension range is zoomed in
├── viewMode: 'layernorm' | 'rmsnorm' | 'both'
├── data: {
│   ├── layers: Array<{
│   │   ├── layerIndex: number
│   │   ├── preNorm: { mean, var, rms, histogram, hiddenStates }
│   │   ├── postNorm: { mean, var, rms, histogram, hiddenStates }
│   │   ├── gamma: number[]  // γ parameters
│   │   ├── beta: number[]   // β parameters
│   │   └── gradientNorms: number[]  // per training step
│   │   }>,
│   └── tokens: string[]
├── computationStatus: 'idle' | 'loading' | 'computing' | 'ready' | 'error'
├── viewConfig: {
│   ├── showPreNormVectors: boolean
│   ├── showPostNormVectors: boolean
│   ├── showConnectionLines: boolean
│   └── colorMode: 'token' | 'magnitude' | 'direction'
├── actions: {
│   ├── loadModel(modelId: string)
│   ├── runInference(text: string)
│   ├── selectLayer(layerIndex: number)
│   ├── selectToken(tokenIndex: number | null)
│   ├── tweakGamma(layerIndex: number, dimIndex: number, value: number)
│   ├── tweakBeta(layerIndex: number, dimIndex: number, value: number)
│   └── toggleViewConfig(key: string)
}
```

---

## 🎯 PART 6: IMPLEMENTATION PRIORITY & PHASING

| Phase | Features | Timeline | Key Deliverable |
|-------|----------|----------|-----------------|
| **Phase 1: Core Inference** | Transformers.js setup, hidden state extraction, basic statistics display (Zone A) | Week 1-2 | Working model inference in browser with pre/post norm statistics |
| **Phase 2: 3D Vector Field** | React Three Fiber sphere + vector visualization (Zone B), PCA in Pyodide | Week 3-4 | Interactive 3D view of LayerNorm mapping |
| **Phase 3: Parameters & Math** | γ/β bar charts (Zone C), KaTeX formula panel (Zone D) | Week 5-6 | Complete educational experience |
| **Phase 4: Diagnostics** | Ablation, Pre-LN vs Post-LN, RMSNorm comparison (Zone E) | Week 7-8 | Research-grade analysis tools |
| **Phase 5: Polish** | Animation polish, performance optimization, fallback handling | Week 9-10 | Production-ready dashboard |

---

## 📚 References

1. **Transformer Explainer** (Georgia Tech, AAAI 2025) — "An interactive visualization tool designed for non-experts to learn about Transformers through the GPT-2 model. Layer Normalization is visualized as a curved line and a circle with tooltips."

2. **Visual Sandbox** (Stanford, NeurIPS 2024) — "A visual sandbox designed to explore the training dynamics of a small-scale transformer model... gains insights into training dynamics... including those induced by the high curvature of normalization layers."

3. **Geometric Interpretation of LayerNorm vs RMSNorm** (ACL) — "All LLMs naturally operate orthogonal to the uniform vector at inference time."

4. **Helios-Web** (2025) — "An open-source, web-based platform designed for real-time visualization and interactive analysis of massive-scale networks and high-dimensional embeddings using GPU-based rendering."

5. **Pyodide** — "Brings the Python data science stack to the browser... includes NumPy, Pandas, Matplotlib, and SciPy out-of-the-box."

6. **Pre-LN vs Post-LN** (He et al., 2020) — "Pre-LN对梯度下降更加友好，收敛更快，对训练时的超参数如学习率等更加鲁棒."

7. **Transformers.js** (Xenova, 2025) — "Run Transformers directly in your browser, with no need for a server... uses ONNX Runtime to run models in the browser."

8. **TensorSpace.js** — "基于TensorFlow.js、Three.js和Tween.js开发，用于对神经网络进行3D可视化呈现."

9. **RMSNorm vs LayerNorm** — "RMSNorm比LayerNorm少了求均值和减均值两个操作，在大模型里能明显提速."

10. **ECharts-GL** — "An extension pack for Apache ECharts that provides 3D visualization capabilities through WebGL."

11. **Deck.gl** — "GPU-driven framework for rendering massive datasets."

12. **Web Workers + SharedArrayBuffer** — "Allows the web application to use additional CPU processing threads so the user interface remains responsive."

13. **ONNX Runtime Web with WebGPU** — "Ship fast, private, and efficient AI in the browser with ONNX Runtime Web, WebGPU, and a proven pipeline."