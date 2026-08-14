# 🧮 Weight Matrix Decomposer — Complete Implementation Specification

> *"The SVD decomposition of transformer weight matrices is highly interpretable — the primary directions used by the network can be inferred from the SVD."* — SVDInterpretTransformer (ICML 2026) 

---

## 0. Architecture Overview

The Weight Matrix Decomposer is a **client‑side computational visualization engine** that performs real‑time matrix decomposition directly in the browser, then renders the results across four coordinated views. This eliminates server round‑trips for computation while leveraging GPU acceleration for rendering.

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Computation Engine** | Pyodide (CPython → WebAssembly) + NumPy + SciPy | Run full SVD, CUR, Kronecker, Tensor Train decompositions in‑browser with near‑native performance. Pyodide "brings Python runtime and scientific computing libraries—including NumPy, Pandas, Matplotlib, and SciPy—directly in the browser" |
| **3D Rendering** | React Three Fiber (R3F) + Three.js + Drei | Declarative React components for 3D matrix visualization, singular value spectrum, and direction space. R3F "lets you write 3D using React components instead of raw Three.js API" |
| **GPU Acceleration** | Deck.gl (WebGL2/WebGPU) | Render massive weight matrices (768×768 to 4096×4096) with GPU‑powered layers. "deck.gl is a GPU‑powered framework for visual exploratory data analysis of large datasets" |
| **2D Charts** | Plotly.js + Recharts | Interactive heatmaps, bar charts for singular value spectrum, cumulative energy curves. Plotly provides "hover, zoom, 3D surface plots, and export to standalone interactive HTML" |
| **WebGL Math** | gl‑matrix | "Hand‑tuned functions for maximum JavaScript performance" for matrix operations, "designed for WebGL applications with Float32Array storage" |
| **Animation Engine** | Manim Community Edition (server‑side) | 3Blue1Brown‑style mathematical animations rendered on‑demand. "Manim is an engine for precise programmatic animations, designed for creating explanatory math videos" |
| **Linear Algebra (JS)** | Loshu.js | "Capable of matrix decompositions, eigenvalue and vectors, solution and approximation to linear systems" in pure JavaScript, as lightweight fallback |
| **State Management** | Zustand + React Query | Manage decomposition state, caching of SVD results, and synchronization across the 4 coordinated views |
| **Web Workers** | Comlink + Pyodide WebWorker | Offload heavy SVD computation to a background thread so the UI remains responsive at 60fps |
| **Data Transfer** | SharedArrayBuffer | Zero‑copy data transfer between the computation worker and the rendering thread for large matrices |

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REACT UI (Main Thread)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Canvas 1 │  │ Canvas 2 │  │ Canvas 3 │  │ Math Panel   │   │
│  │ (Spectrum│  │(Direction│  │(Compress │  │ (LaTeX Live) │   │
│  │  View)   │  │  View)   │  │ Preview) │  │              │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       │              │              │               │           │
│       └──────────────┴──────────────┴───────────────┘           │
│                          │                                       │
│              ┌───────────┴───────────┐                          │
│              │   Zustand Store       │                          │
│              │  (decompositionState) │                          │
│              └───────────┬───────────┘                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │   Comlink Proxy        │
              │   (Async Communication) │
              └────────────┬────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│              WEB WORKER (Background Thread)                      │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────────────────┐  │
│  │              Pyodide Runtime                               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │  │
│  │  │  NumPy  │  │  SciPy  │  │ Scikit- │  │  Custom     │  │  │
│  │  │  (SVD)  │  │(sparse) │  │  Learn  │  │  (CUR, TT,  │  │  │
│  │  │         │  │         │  │  (NMF)  │  │  Kronecker) │  │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           SharedArrayBuffer (Zero‑Copy Transfer)           │  │
│  │  [Weight Matrix] ←→ [U, Σ, V^T] ←→ [Reconstructed Matrix] │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 1. CANVAS 1: Singular Value Spectrum View

### Implementation Architecture

This view displays the singular value spectrum of a selected weight matrix as an interactive visualization. The design follows the SVDInterpretTransformer approach: "the projection matrices of MLP sublayers admit a natural decomposition into orthogonal, interpretable rank‑1 subspaces".

#### Sub‑Component Tree

```
<SingularValueSpectrum>
  ├── <MatrixSelector>           // Dropdown: Q, K, V, O, FFN1, FFN2 per layer
  ├── <SpectrumChart>            // Plotly.js bar + line chart
  │   ├── <SingularValueBar>     // One bar per σᵢ, color by cumulative energy
  │   ├── <CumulativeEnergyLine> // Overlaid line: Σσ² / Σσ²_total
  │   └── <HoverTooltip>         // Shows σᵢ value, individual %, cumulative %
  ├── <RankSlider>               // Interactive k slider: 1 … max_rank
  │   ├── <EnergyThresholdMark>  // Dashed line at 95% energy
  │   └── <RankPreviewBadge>     // "Rank 64 → 98.3% energy retained"
  ├── <SingularVectorTable>      // Top-5 singular vectors with word cloud
  └── <DecompositionMetadata>    // Computation time, matrix dimensions, norm
```

#### Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Spectrum Chart | **Plotly.js** `Bar` + `Scatter` overlay | GPU‑accelerated WebGL rendering, built‑in hover/zoom/export, handles 1000+ bars smoothly |
| Bar Colors | **D3‑interpolate** `interpolateRdBu` | Color gradient from blue (σ₁) to red (σₙ) showing importance decay |
| Rank Slider | **Radix UI** `Slider` + **Framer Motion** | Accessible, animated, with spring physics for smooth value transitions |
| Word Cloud | **React‑Wordcloud** (D3‑based) | Visualize top tokens aligned with each singular vector direction |

#### Data Flow

1. User selects weight matrix (e.g., "Layer 3 — Query Projection W_Q")
2. React component calls `decompositionWorker.computeSVD(matrixId)` via Comlink
3. Pyodide Web Worker loads the matrix from a `SharedArrayBuffer`, runs `np.linalg.svd(W, full_matrices=False)`
4. Returns `{U_shape, S_array, Vt_shape, compute_time_ms}` via Comlink proxy
5. Zustand store updates → all 4 canvases re‑render reactively
6. Spectrum chart renders σ₁…σₙ as descending bars with cumulative energy overlay

#### Key Interaction: Energy‑Threshold Rank Selection

Research shows "small singular values matter" — "our findings highlight the overlooked importance of the low end of the spectrum and provide theoretical and practical guidance for SVD‑based pruning and compression". The user drags the rank slider; the chart updates:

- **Bars to keep** (σ₁…σ_k): full color opacity
- **Bars to discard** (σ_{k+1}…σ_n): grayed out, 30% opacity
- **Cumulative energy line**: vertical dashed line at k, showing "95.3% energy retained"
- **Reconstruction error**: Frobenius norm ‖W − W_k‖_F displayed live

---

## 2. CANVAS 2: SVD Direction Interpreter

### Implementation Architecture

This view visualizes the semantic meaning of each singular vector direction. Based on the SVDInterpretTransformer finding: "each unique term in a corpus is assigned to a row… you take the SVD" revealing interpretable directions.

#### Sub‑Component Tree

```
<SVDirectionInterpreter>
  ├── <DirectionSelector>        // Which singular vector (1…k) to explore
  ├── <Direction3DScatter>       // R3F 3D scatter: all tokens projected onto U_i
  │   ├── <TokenSphere>          // One sphere per vocabulary token
  │   ├── <DirectionArrow>       // Arrow showing the singular vector direction
  │   └── <OrbitControls>        // R3F‑Drei camera controls
  ├── <TokenRankingTable>        // Top-20 tokens most aligned with this direction
  ├── <DirectionWordCloud>       // Word cloud visualization
  └── <SemanticLabel>            // Auto‑generated label: "This direction captures verb tense"
```

#### Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| 3D Scatter | **React Three Fiber** + **@react-three/drei** `Sphere`, `Line`, `Text` | "R3F lets you write 3D using React components" with GPU‑accelerated rendering via Three.js |
| Scatter Layout | **UMAP** (via Pyodide) or **t‑SNE** | Dimensionality reduction for token embedding visualization in 3D space |
| Word Cloud | **D3‑cloud** layout | Classic word cloud with font size proportional to alignment score |
| Semantic Label | **LLM‑generated** (server‑side) | Feed top‑20 tokens to GPT‑4o mini for interpretable label generation |

#### Data Flow

1. User selects singular vector index `i` (1 ≤ i ≤ k)
2. Worker computes the projection: `scores_j = |U_i · E_j|` for each token embedding E_j
3. Top‑50 tokens returned to main thread
4. 3D scatter: each token is a sphere; position = (scores_j, x_j, y_j) where x,y are UMAP projections
5. Direction arrow: shows the orientation of the singular vector in embedding space

---

## 3. CANVAS 3: Compression Preview — "What If?" Studio

### Implementation Architecture

This is the most powerful and novel view — it lets researchers **interactively explore the compression landscape** by adjusting decomposition parameters and seeing real‑time reconstruction quality. Directly connected to your **60+ compression methods** (A3, ERC‑SVD, AdaSVD, CUR, Kronecker, TT, etc.).

#### Sub‑Component Tree

```
<CompressionPreview>
  ├── <CompressorSelector>       // Dropdown: All registered compressors
  ├── <ParameterPanel>           // Dynamic params based on selected compressor
  │   ├── <RankSlider>           // For SVD‑based: k = 1…min(m,n)
  │   ├── <EnergyThreshold>      // For ERC‑SVD: 0.90 … 0.999
  │   ├── <LeverageSampling>     // For CUR: uniform vs. leverage score
  │   ├── <FactorDims>           // For Kronecker: m₁,m₂, n₁,n₂
  │   └── <TTRanks>              // For Tensor Train: r₁, r₂, …
  ├── <ReconstructionGrid>       // Side‑by‑side: Original vs. Reconstructed
  │   ├── <OriginalMatrix>       // Deck.gl BitmapLayer — original W
  │   ├── <ReconstructedMatrix>  // Deck.gl BitmapLayer — W_approx
  │   └── <DifferenceMatrix>     // Deck.gl BitmapLayer — |W − W_approx|
  ├── <CompressionStats>         // Live stats panel
  │   ├── <ParameterReduction>   // "589,824 → 98,176 params (83.4% reduction)"
  │   ├── <FrobeniusError>       // ‖W − W_approx‖_F / ‖W‖_F
  │   ├── <SpeedUpEstimate>      // Estimated FLOPs reduction
  │   └── <AccuracyPrediction>   // ML‑predicted accuracy impact
  ├── <LayerComparison>          // How does this layer compare to others?
  └── <ApplyButton>              // "Apply This Compression to the Model"
```

#### Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Matrix Visualization | **Deck.gl** `BitmapLayer` or custom **R3F** instanced mesh | "deck.gl is a GPU‑powered framework… WebGL2/WebGPU‑based visualization of large data sets" — can render 768×768 matrices as pixel‑perfect grids at 60fps |
| Difference Matrix | **R3F** shader material with custom GLSL | GPU‑accelerated element‑wise difference computation and color mapping |
| Stats Panel | **Recharts** `RadialBar` + **Framer Motion** animated numbers | Already in your stack; spring‑animated number transitions |
| Accuracy Prediction | **Pyodide** + small ONNX model | Run a lightweight regression model that predicts accuracy impact from (rank_ratio, layer_type, original_norm) |
| Apply Button | Your existing **`launcher`** API | Pre‑fills experiment config with compression parameters |

#### Data Flow for "What If?" Interaction

1. User selects "CUR decomposition" from the compressor dropdown
2. Worker computes CUR: column selection via leverage scores → `C`, row selection → `R`, pseudo‑inverse → `U`
3. `W_approx = C @ U @ R` computed in Pyodide
4. Original W, W_approx, and element‑wise difference sent to main thread via SharedArrayBuffer
5. Deck.gl renders all three matrices as tiled heatmaps with synchronized zoom/pan
6. Stats panel updates reactively from the store

#### Key Interaction: Real‑Time Reconstruction

As the user drags the rank slider, the reconstruction updates at **30fps** using a debounced Pyodide call. The implementation uses:

- **Throttled SVD**: If the user is sliding, compute only for the final value after a 200ms debounce
- **Cached intermediate results**: For CUR, cache the column/row indices; only recompute U
- **Web Worker isolation**: Heavy computation never blocks the main thread

---

## 4. MATH PANEL: Live LaTeX + Numerical Computation

### Implementation Architecture

This panel renders the mathematical formulas for the selected decomposition in real‑time LaTeX, with live numerical values that update as the user interacts with the visualization.

#### Sub‑Component Tree

```
<MathPanel>
  ├── <DecompositionFormula>     // Live LaTeX: W = U Σ V^T
  │   ├── <MatrixNotation>       // Dimensional annotations: (768×768)
  │   └── <TruncationIndicator>  // Highlights which parts are kept/discarded
  ├── <StepByStepComputation>    // Animated computation walkthrough
  │   ├── <Step1_MatrixMult>     // Q^T K step
  │   ├── <Step2_Scale>          // ÷ √d_k
  │   ├── <Step3_Softmax>        // Row‑wise softmax
  │   └── <Step4_WeightedSum>    // × V
  ├── <NumericalInspector>       // Live values at hover position
  └── <ManimAnimation>           // On‑demand 3Blue1Brown‑style video
```

#### Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| LaTeX Rendering | **KaTeX** (fast) or **MathJax 3** (comprehensive) | Render complex matrix equations with dimensional annotations |
| Numerical Inspector | **React `useMemo`** + **Framer Motion** | Compute values reactively; animate number changes |
| Step Animation | **Framer Motion** `AnimatePresence` + custom SVG | Show matrix multiplication as animated blocks |
| Manim Video | **Manim Community Edition** (server‑side) | "Battle‑tested code examples and guidelines for creating mathematical animations"; render server‑side, stream as MP4 |

#### Implementation Notes for Math Panel

**LaTeX Live Rendering**: KaTeX renders the formula `W = U Σ V^T` with color‑coded dimensions. When the user hovers over a singular value bar in Canvas 1, the corresponding σᵢ highlights in the LaTeX. When they change the rank k, the truncation indicator updates to show `W_k = U_k Σ_k V_k^T`.

**Step‑by‑Step Animation**: Uses SVG `<rect>` elements animated with Framer Motion's `layoutId` for seamless transitions. Each step highlights the current operation and shows intermediate numerical results.

**Manim Animation**: When the user clicks "Watch Explanation," a request is sent to a server endpoint that runs Manim CE with a pre‑written SVD explanation scene. The rendered MP4 is streamed back. "Manim is an engine for precise programmatic animations, designed for creating explanatory math videos".

---

## 5. ANALYTICS PANEL: Per‑Layer Rank Profile & Compression Quality

### Implementation Architecture

This panel provides a bird's‑eye view of compressibility across all layers, helping researchers identify which layers are most amenable to compression.

#### Sub‑Component Tree

```
<CompressionAnalytics>
  ├── <LayerRankProfile>         // Bar chart: optimal rank per layer
  │   ├── <EnergyThresholdLine>  // Horizontal line at 95% energy
  │   └── <LayerBar>             // One bar per layer, height = rank needed for 95% energy
  ├── <SingularValueDecayChart>  // Overlaid decay curves for all layers
  ├── <CompressionQualityMatrix> // Heatmap: layer × rank → predicted accuracy
  ├── <CompressorComparison>     // Radar chart: compare SVD vs. CUR vs. Kronecker vs. TT
  └── <RecommendationEngine>     // "We recommend compressing Layers 3‑6 with ERC‑SVD (rank=64)"
```

#### Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Rank Profile | **Recharts** `BarChart` with custom shape | Already in your stack; clean bar chart with threshold line |
| Decay Curves | **Plotly.js** `Scatter` multiple traces | GPU‑accelerated for many overlaid curves |
| Quality Matrix | **Deck.gl** `HeatmapLayer` or Plotly `Heatmap` | Visualize the 2D accuracy landscape |
| Radar Chart | **Recharts** `RadarChart` | Compare compression methods across multiple dimensions |
| Recommendations | **Server‑side LLM** | Generate natural language recommendations from analytics data |

---

## 6. CONTROLS: Compressor Selector & Parameter Configuration

### Implementation Architecture

The control panel provides unified access to all 60+ compressors in your library, with dynamic parameter forms that adapt to each method.

#### Sub‑Component Tree

```
<CompressorControls>
  ├── <CompressorSelector>       // Searchable dropdown: all registered compressors
  │   ├── <CompressorCategory>   // Grouped: SVD‑based, CUR‑based, Pruning, Quantization
  │   └── <CompressorSearch>     // Type to filter
  ├── <DynamicParameterForm>     // Renders different params based on compressor
  │   ├── <SVDRankSlider>        // For A3, ERC, AdaSVD: rank slider
  │   ├── <CURSamplingMethod>    // For CUR: uniform vs. leverage
  │   ├── <KroneckerDims>        // For Kronecker: factor dimensions
  │   ├── <TTRankConfig>         // For TT: rank per bond
  │   └── <NMFConfig>            // For NMF: n_components, init method
  ├── <ComputeButton>            // "Decompose" — triggers Pyodide computation
  └── <PresetSelector>           // Quick presets: "Fast", "Accurate", "Extreme"
```

#### Implementation Notes

The `DynamicParameterForm` uses a registry pattern: each compressor exports a `CompressorConfigSchema` that describes its parameters (name, type, min, max, default). The form renders appropriate inputs for each parameter type.

```typescript
// Conceptual type definition
interface CompressorConfigSchema {
  method: string;
  parameters: {
    name: string;
    type: 'slider' | 'dropdown' | 'number' | 'toggle';
    min?: number;
    max?: number;
    step?: number;
    options?: { label: string; value: string }[];
    default: any;
  }[];
}
```

---

## 7. CROSS‑PLATFORM INTEGRATION: One‑Click Compress Pipeline

### Implementation Architecture

The most powerful feature: directly connecting the visualization to your compression infrastructure. When a user finds a promising compression configuration, they can apply it to the actual model with one click.

#### Integration Points

| Integration | Endpoint / Module | Data Flow |
|-------------|-------------------|-----------|
| **Compressor Registry** | `framework.core.registry.COMPRESSOR_REGISTRY` | Read available compressors and their parameter schemas |
| **Apply Compression** | `POST /api/runs/start-compress/` | Send compression config, model ID, layer selection |
| **Experiment Monitor** | `ws/experiments/{run_id}/` | WebSocket stream for real‑time progress |
| **Model Versioning** | `model_versioning` app | Auto‑create version with compressed model |
| **Benchmark** | `gpu_benchmark` app | Run MLPerf‑style benchmark on compressed model |
| **Leaderboard** | `leaderboards` app | Submit compressed model performance to leaderboard |

#### Data Flow for One‑Click Compression

1. User configures compression parameters in the dashboard (e.g., ERC‑SVD, rank=64, energy_threshold=0.95)
2. Clicks **"Apply This Compression to the Model"**
3. Frontend calls `POST /api/runs/start-compress/` with:
   ```json
   {
     "model_name": "bert-base-uncased",
     "compressor": "erc",
     "compressor_config": { "erc_allowed_error": 0.05 },
     "target_layers": ["encoder.layer.3.attention.self.query"],
     "experiment_name": "Compress BERT Layer 3 Query — ERC‑SVD rank 64"
   }
   ```
4. Backend creates a `TrainingRun` with `job_type='compression'`
5. WebSocket streams progress: "Computing SVD…", "Reconstructing weights…", "Benchmarking…"
6. On completion: before/after comparison appears in the dashboard automatically

---

## 8. PERFORMANCE OPTIMIZATION STRATEGIES

### Matrix Size Tiers

| Matrix Size | Computation Strategy | Rendering Strategy |
|-------------|---------------------|-------------------|
| ≤ 512×512 | Pyodide SVD in Web Worker (~50ms) | Deck.gl BitmapLayer, 60fps |
| 512×512 – 2048×2048 | Pyodide SVD in Web Worker (~500ms), show progress bar | Deck.gl with Level‑of‑Detail, 30fps |
| > 2048×2048 | Server‑side SVD via API, stream results | Deck.gl with tiled rendering, progressive loading |

### Caching Strategy

- **LRU Cache** (Zustand middleware): cache the last 20 SVD results with key = `(matrix_hash, rank)`
- **Pre‑computation**: when a model is loaded, pre‑compute SVD for all linear layers in a Web Worker pool
- **Incremental loading**: load singular values first (fast), then singular vectors on‑demand

### Memory Management

- **SharedArrayBuffer**: transfer weight matrices between threads without copying
- **Float32Array**: all matrix data stored as Float32 (not Float64) to halve memory usage
- **Garbage Collection**: explicitly nullify large arrays when switching views

---

## 9. IMPLEMENTATION FILE STRUCTURE

```
src/features/visualizations/weight-decomposer/
├── index.tsx                          # Main WeightDecomposer page
├── WeightDecomposerDashboard.tsx      # Layout with 4‑zone grid
├── components/
│   ├── canvas/
│   │   ├── SingularValueSpectrum.tsx  # Canvas 1: Spectrum view
│   │   ├── DirectionInterpreter.tsx   # Canvas 2: Direction view (R3F)
│   │   └── CompressionPreview.tsx     # Canvas 3: Compression preview (Deck.gl)
│   ├── panels/
│   │   ├── MathPanel.tsx              # LaTeX + numerical panel
│   │   ├── AnalyticsPanel.tsx         # Per‑layer analytics
│   │   └── CompressorControls.tsx     # Compressor selector + params
│   ├── shared/
│   │   ├── MatrixSelector.tsx         # Layer/weight selector
│   │   ├── DecompositionWorker.ts     # Comlink interface to Pyodide worker
│   │   └── SharedArrayBufferManager.ts # SAB lifecycle management
│   └── integration/
│       ├── CompressPipelineButton.tsx  # "Apply Compression" button
│       └── ExperimentMonitor.tsx       # Real‑time compression progress
├── workers/
│   ├── svd.worker.ts                  # Pyodide SVD computation worker
│   ├── cur.worker.ts                  # CUR decomposition worker
│   ├── kronecker.worker.ts            # Kronecker factorization worker
│   └── tensor_train.worker.ts         # TT decomposition worker
├── hooks/
│   ├── useDecomposition.ts            # Main hook: triggers computation, manages cache
│   ├── usePyodideWorker.ts            # Comlink wrapper for Pyodide worker
│   └── useCompressionPipeline.ts      # Hook for "Apply Compression" flow
├── stores/
│   └── decompositionStore.ts          # Zustand store for all decomposition state
├── utils/
│   ├── svdCalculations.ts             # Pure JS SVD utilities (fallback)
│   ├── compressionMetrics.ts          # Frobenius norm, energy computation
│   └── colorScales.ts                 # Color mapping for matrix visualization
└── types/
    └── decomposition.ts               # TypeScript types for all decomposition results
```

---

## 10. KEY DEPENDENCIES (package.json additions)

```json
{
  "dependencies": {
    "pyodide": "^0.26.0",
    "comlink": "^4.4.0",
    "@react-three/fiber": "^8.15.0",
    "@react-three/drei": "^9.92.0",
    "three": "^0.160.0",
    "deck.gl": "^9.0.0",
    "@deck.gl/layers": "^9.0.0",
    "plotly.js-dist-min": "^2.30.0",
    "react-plotly.js": "^2.6.0",
    "gl-matrix": "^3.4.0",
    "katex": "^0.16.0",
    "framer-motion": "^10.16.0",
    "react-wordcloud": "^1.2.0"
  },
  "devDependencies": {
    "@types/three": "^0.160.0",
    "worker-loader": "^3.0.8"
  }
}
```

---

## 📊 Summary: Implementation Priority & Effort

| # | Component | Effort (days) | Dependencies | Key Risk |
|---|-----------|---------------|--------------|----------|
| 1 | Pyodide SVD Worker | 3 | Pyodide, Comlink, SharedArrayBuffer | Pyodide initialization time (~3s first load) |
| 2 | Singular Value Spectrum (Canvas 1) | 2 | Plotly.js, Recharts | Handling 4096+ singular values smoothly |
| 3 | SVD Direction Interpreter (Canvas 2) | 4 | R3F, Drei, UMAP (Pyodide) | 3D performance with many token spheres |
| 4 | Compression Preview (Canvas 3) | 5 | Deck.gl, custom GLSL shaders | Large matrix rendering performance |
| 5 | Math Panel | 2 | KaTeX, Framer Motion | LaTeX rendering performance |
| 6 | Analytics Panel | 3 | Recharts, Plotly.js | Multi‑layer data aggregation |
| 7 | Compressor Controls | 2 | Radix UI, Framer Motion | Dynamic form generation from schema |
| 8 | Cross‑Platform Integration | 3 | Your experiment launcher, WebSocket | API compatibility across all compressors |
| 9 | CUR, Kronecker, TT Workers | 4 (each) | Pyodide, custom algorithms | Algorithm correctness for all decompositions |
| 10 | Caching & Performance | 2 | Zustand middleware, LRU | Cache invalidation on model change |

**Total estimated effort: ~35‑40 days for complete implementation of all decomposition methods.**

The Weight Matrix Decomposer transforms your compression library from a set of black‑box algorithms into an interactive, educational, and research‑grade exploration tool — where every singular value, every CUR column selection, and every Kronecker factorization is visible, understandable, and directly actionable.