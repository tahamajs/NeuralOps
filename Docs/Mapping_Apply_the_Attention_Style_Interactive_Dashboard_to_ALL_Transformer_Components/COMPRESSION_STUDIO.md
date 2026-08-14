# 🏗️ Compression Studio — Complete Implementation Specification

## "Interactive Model Surgery" for the Attention Visualization Dashboard

This document provides an **exhaustive, production‑grade implementation blueprint** for the Compression Studio — a full‑featured interactive model compression environment integrated into the NeuralOps Attention Dashboard. Every section specifies the technologies, architectural patterns, component hierarchies, data flow designs, rendering strategies, and integration touchpoints needed to build this system from scratch.

---

## Table of Contents

1. [Architectural Overview & Technology Stack](#1-architectural-overview--technology-stack)
2. [Zone‑by‑Zone Implementation](#2-zone‑by‑zone-implementation)
3. [Zone 1: Model Architecture Map — The Interactive Tree Canvas](#3-zone-1-model-architecture-map)
4. [Zone 2: Compression Impact Heatmap — The WebGL/WebGPU Layer](#4-zone-2-compression-impact-heatmap)
5. [Zone 3: Live Math Panel — The LaTeX Computation Engine](#5-zone-3-live-math-panel)
6. [Zone 4: Controls — The Compression Parameter Orchestrator](#6-zone-4-controls)
7. [Zone 5: Cross‑Platform Integration — The Action Pipeline](#7-zone-5-cross‑platform-integration)
8. [State Management Architecture](#8-state-management-architecture)
9. [Performance & Rendering Strategy](#9-performance--rendering-strategy)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Architectural Overview & Technology Stack

The Compression Studio is not a single component — it is a **multi‑layer, multi‑runtime application** composed of five tightly coupled zones that share state through a centralized state machine. The architecture draws from: **OpenAI's circuit_sparsity Streamlit dashboard** (parameterized pruning sweep exploration), **DL‑Playground's** drag‑and‑drop neural network editor with visual architecture construction, **Penzai + Treescope's** interactive tensor visualization from Google DeepMind, **React Flow's** node‑graph rendering substrate (used by AgentForge and FlowiseAI), **deck.gl's** GPU‑powered large‑scale data visualization framework, and **GPT2‑WebGL's** browser‑based transformer implementation with WebGL2 matrix visualization.

### 1.1 Technology Stack Table

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI Framework** | React 18+ with TypeScript | Component rendering, hooks, context |
| **Build Tool** | Vite 5+ | Fast HMR, optimized builds, code splitting |
| **3D Rendering Engine** | React Three Fiber (@react‑three/fiber 9.x) | 3D model architecture visualization, WebGL/WebGPU rendering |
| **3D Utilities** | @react‑three/drei | OrbitControls, Text, Line, shaders, environment maps |
| **2D Graph Editor** | React Flow 12.x (xyflow) | Interactive node‑based model architecture tree editor |
| **GPU Data Visualization** | deck.gl 9.x + @deck.gl/react | Large‑scale compression impact heatmap, scatter plots, layers |
| **Math Rendering** | KaTeX (katex 0.16+) | Client‑side LaTeX rendering for compression formulas |
| **State Management** | Zustand 5.x + XState 5.x | Global application state + finite state machines for compression workflows |
| **Server State** | TanStack Query 5.x (React Query) | API caching, optimistic updates, background refetching |
| **Animation** | Framer Motion 11.x | Component transitions, layout animations, gesture handling |
| **Drag & Drop** | @dnd‑kit/core + @dnd‑kit/sortable | Layer/head reordering, compression priority drag |
| **Color Scales** | d3‑scale + d3‑interpolate | Scientific color maps (viridis, magma, cividis) |
| **Python Runtime (Browser)** | Pyodide 0.27+ (CPython 3.12 compiled to WASM) | Client‑side SVD computation, matrix operations, validation |
| **ML Inference (Browser)** | ONNX Runtime Web (ort‑web 1.19+) + WebGPU backend | Browser‑side model inference for before/after comparison |
| **Web Worker** | Comlink | Off‑main‑thread SVD and compression computation |
| **WebSocket** | Native WebSocket + ReconnectingWebSocket | Real‑time compression progress from Django backend |
| **Testing** | Vitest + Testing Library + Playwright | Unit, integration, visual regression, E2E |
| **CSS** | Tailwind CSS 3.4+ + CSS Modules | Utility‑first styling with scoped component styles |

### 1.2 Browser‑Side Compute Architecture

The Compression Studio performs **actual SVD computation in the browser** using a dual‑approach strategy:

1. **Small matrices (< 1000×1000):** Compute via **Pyodide** (NumPy/SciPy running in WebAssembly). Pyodide compiles CPython 3.12 to WebAssembly, enabling `numpy.linalg.svd()` directly in the browser without server round‑trips. This is ideal for per‑layer weight matrices and singular value spectrum visualization.

2. **Large matrices (≥ 1000×1000):** Offload to a **Web Worker** using Comlink for RPC‑style communication, or delegate to the Django backend via REST API. The Web Worker runs the SVD using either Pyodide or a pure JavaScript implementation, keeping the main thread responsive.

3. **GPU‑accelerated visualization:** The compression impact heatmap renders via **deck.gl's WebGL2/WebGPU layer system**, which can handle millions of data points (weight entries, activation differences) at 60fps. GPT2‑WebGL demonstrates that full transformer model matrices can be loaded as WebGL textures and visualized in‑browser.

### 1.3 Project Directory Structure

```
src/features/compression-studio/
├── index.tsx                          # Entry point, lazy‑loaded by router
├── CompressionStudio.tsx              # Root layout orchestrator (5 zones)
├── context/
│   ├── CompressionContext.tsx          # React Context provider
│   └── CompressionStudioProvider.tsx   # Combines Zustand + XState + React Query
├── store/
│   ├── compressionStore.ts            # Zustand store (model, compressor, params, budget)
│   ├── compressionMachine.ts          # XState state machine (idle→analyzing→compressing→comparing)
│   └── selectors.ts                   # Derived state selectors (effective rank, FLOPs, etc.)
├── hooks/
│   ├── useCompressionAnalysis.ts      # TanStack Query hook for backend analysis
│   ├── useBrowserSVD.ts               # Pyodide/Web Worker SVD computation
│   ├── useCompressionProgress.ts      # WebSocket subscription for live progress
│   ├── useModelArchitecture.ts        # Fetch and parse model architecture tree
│   ├── useParameterCount.ts           # Real‑time parameter count computation
│   └── useBudgetConstraint.ts         # Budget constraint solver
├── workers/
│   ├── svd.worker.ts                  # Web Worker for SVD computation
│   └── compression.worker.ts          # Web Worker for compression simulation
├── zone1-architecture-map/
│   ├── ModelArchitectureTree.tsx       # React Flow‑based tree view
│   ├── ArchitectureNode.tsx            # Custom React Flow node (layer/head/neuron)
│   ├── CompressionNodeOverlay.tsx      # Overlay showing compression options
│   ├── BeforeAfterSplitView.tsx        # Side‑by‑side original vs compressed
│   └── hooks/
│       └── useArchitectureLayout.ts    # Dagre layout computation
├── zone2-impact-heatmap/
│   ├── CompressionImpactHeatmap.tsx    # deck.gl‑based impact visualization
│   ├── ImpactHeatmapLayer.ts           # Custom deck.gl layer
│   ├── ImpactColorScale.ts             # d3 color scale for impact
│   └── ImpactTooltip.tsx               # Hover tooltip with per‑parameter detail
├── zone3-math-panel/
│   ├── LiveMathPanel.tsx               # LaTeX formula renderer
│   ├── CompressionFormulaDisplay.tsx   # Shows selected compressor's formula
│   ├── ParameterCountComparison.tsx    # Before/after parameter count bar chart
│   └── FLOPsEstimator.tsx             # Real‑time FLOPs estimation
├── zone4-controls/
│   ├── CompressorSelector.tsx          # Dropdown with all 60+ compressors
│   ├── ParameterSliders.tsx            # Rank/ratio/sparsity/bit‑width sliders
│   ├── BudgetConstraintInput.tsx       # "Keep model under X MB" input
│   ├── CompressionPresetSelector.tsx    # "Fast", "Accurate", "Balanced" presets
│   └── ApplyButton.tsx                 # "Compress Now" action button
├── zone5-cross-platform/
│   ├── OneClickApply.tsx               # Bridge to your compression pipeline
│   ├── BeforeAfterLeaderboard.tsx       # Submit compressed model to leaderboard
│   └── CompressionHistoryPanel.tsx      # History of compression runs
└── components/
    ├── SingularValueSpectrum.tsx        # Interactive singular value plot (Recharts)
    ├── WeightDistributionChart.tsx      # Weight histogram before/after
    ├── CompressionDiffView.tsx          # Side‑by‑side output comparison
    ├── ModelStatsCard.tsx               # Parameter count, FLOPs, memory usage
    └── CompressionProgressBar.tsx       # Animated progress bar for compression job
```

---

## 2. Zone‑by‑Zone Implementation

### 3. Zone 1: Model Architecture Map — The Interactive Tree Canvas

#### 3.1 Core Concept

The Model Architecture Map is a **React Flow‑based interactive tree** that renders the complete model architecture as a navigable graph. Each node represents a component (attention head, FFN neuron, embedding dimension, normalization layer). The user clicks any component to inspect its parameters, importance scores, and available compression options. A **split‑view mode** shows the original architecture side‑by‑side with a **live preview** of the compressed architecture.

#### 3.2 Technology & Framework Choices

**Primary rendering substrate: React Flow 12.x (xyflow)**

React Flow is chosen as the rendering engine because it provides:
- Built‑in node dragging, zooming, panning, and minimap
- Custom node rendering with React components
- Edge (connection) rendering with labels and styling
- Event handling for node click, hover, selection
- Programmatic layout control

Multiple production AI workflow editors use React Flow as their foundation, including the Workflow Designer UI for Giselle (which uses Zustand for state management alongside React Flow), AgentForge's pipeline editor, and FlowiseAI's agent flow editor.

**Layout engine: Dagre**

Dagre is used to compute a hierarchical (top‑to‑bottom or left‑to‑right) layout for the model architecture tree. The layout is computed once when the model loads and recalculated when layers are pruned or compressed. React Flow provides a `useNodesInitialized` hook that integrates with Dagre's positioning.

**3D complement: React Three Fiber (r3f)**

For an alternative "immersive" view, the same architecture data can be rendered in 3D using React Three Fiber. The `r3f-nn-visualizer` project demonstrates interactive neural network visualization with tf.js and react‑three‑fiber, including live demos with layer‑by‑layer inspection. The Vercel AI SDK's json‑render project further shows how structured JSON specifications (model architecture) can be rendered as 3D scenes with React Three Fiber components.

#### 3.3 Component Hierarchy

```
ModelArchitectureTree (React Flow wrapper)
├── ReactFlow
│   ├── ArchitectureNode (custom node — per layer/head/neuron)
│   │   ├── NodeBody (colored rectangle with label)
│   │   ├── ImportanceIndicator (circular gauge showing importance score)
│   │   ├── CompressionBadge (icon indicating compressibility)
│   │   └── NodeTooltip (hover tooltip with detailed stats)
│   ├── CompressionEdge (custom edge — connections between layers)
│   │   ├── EdgePath (SVG path, thickness = parameter count)
│   │   └── EdgeLabel (connection type: "Q", "K", "V", "FFN", "Residual")
│   └── Minimap (small overview in corner)
├── BeforeAfterSplitView
│   ├── OriginalArchitecturePanel (left side)
│   └── CompressedArchitecturePanel (right side, updates live)
└── NodeInspector (detail panel, slides in from right)
    ├── ParameterTable (weight shape, bias, count)
    ├── ImportanceScoreGauge (circular gauge, 0-100)
    ├── SingularValueSpectrum (small inline chart)
    └── CompressionOptions (checkboxes/sliders for this node)
```

#### 3.4 Data Flow

```
Django Backend
    │
    │ GET /api/models/{id}/architecture/
    ▼
useModelArchitecture() hook (TanStack Query)
    │
    │ Fetches architecture JSON: { layers: [{ type, heads, neurons, params }] }
    ▼
useArchitectureLayout() hook (Dagre)
    │
    │ Computes x, y positions for each node based on hierarchy
    ▼
compressionStore (Zustand)
    │
    │ Stores: selectedCompressor, compressorParams, compressedArchitecture
    ▼
ModelArchitectureTree renders nodes + edges
    │
    │ User clicks a node → NodeInspector opens
    ▼
NodeInspector shows per‑component compression options
    │
    │ User adjusts sliders → Zustand updates → compressedArchitecture recalculated
    ▼
BeforeAfterSplitView re‑renders with updated compressed view
```

#### 3.5 Key Implementation Details

**Custom React Flow Node (ArchitectureNode)**

The custom node uses React Flow's `Handle` components for connection points and renders a styled card with:
- Background color gradient based on importance score (green=important, red=prunable)
- A circular gauge (SVG arc) showing the importance percentage
- An icon badge indicating the compression method applied (scissors for pruning, sigma for SVD, etc.)
- The node is **resizable** — larger nodes have more parameters

**Node selection behavior:**
- Single click → opens `NodeInspector` with per‑component details
- Double click → zooms into that node (expands to show sub‑components like individual heads)
- Shift+click → adds to multi‑selection for batch compression operations
- Right click → context menu with "Compress this component," "Exclude from compression," "Set as compression anchor"

**Before/After Split View**

The split view uses two synchronized React Flow instances (locked viewport). The left shows the original architecture; the right shows a live preview of the compressed architecture. Differences are highlighted:
- **Green outline:** component unchanged
- **Orange outline:** component compressed
- **Red outline with strikethrough:** component pruned

The split view is draggable (the divider can be moved). When the user hovers over a compressed component, a tooltip shows the **delta**: "Parameters: 589K → 98K (83% reduction), FLOPs: 2.3M → 0.4M."

**Model Architecture JSON Schema**

The backend serves architecture data in a standardized JSON format:

```json
{
  "model_name": "bert-base-uncased",
  "total_parameters": 109482240,
  "total_flops_estimate": 2.3e10,
  "layers": [
    {
      "layer_index": 0,
      "type": "encoder",
      "components": [
        {
          "type": "attention",
          "heads": [
            {
              "head_index": 0,
              "parameters": {"q_weight": [768, 64], "k_weight": [768, 64], "v_weight": [768, 64]},
              "importance_score": 0.87,
              "stability_score": 0.92,
              "similarity_score": 0.45,
              "svd_spectrum": [12.3, 8.7, 5.2, 3.1, 1.8, 0.9, ...]
            }
          ]
        },
        {
          "type": "ffn",
          "parameters": {"intermediate_weight": [3072, 768], "output_weight": [768, 3072]},
          "importance_score": 0.65
        }
      ]
    }
  ]
}
```

---

### 4. Zone 2: Compression Impact Heatmap — The WebGL/WebGPU Layer

#### 4.1 Core Concept

The Compression Impact Heatmap shows a **layer‑by‑layer, parameter‑by‑parameter visualization** of how a selected compression strategy will affect the model. Each "cell" in the heatmap represents a weight matrix entry; the **color intensity** encodes the **relative change** in that entry after compression (original value minus compressed value, normalized). Users can zoom into individual weight matrices, hover to see exact before/after values, and compare different compression strategies side‑by‑side.

#### 4.2 Technology & Framework Choices

**Primary rendering: deck.gl 9.x**

deck.gl is chosen because it is "the definitive high‑performance, GPU‑powered framework for rendering massive datasets on the web, specifically addressing the challenge of visualizing millions of data points". The compression impact heatmap may need to render weight matrices with hundreds of thousands of entries — deck.gl's `BitmapLayer`, `HeatmapLayer`, and custom `CompositeLayer` can handle this at 60fps using WebGL2/WebGPU.

deck.gl also provides a React integration (`@deck.gl/react`) that wraps deck.gl into React's lifecycle, including a `DeckGL` React component and hooks for state management.

**Alternative: GPT2‑WebGL approach**

GPT2‑WebGL demonstrates that full transformer weight matrices can be loaded as WebGL2 textures and rendered in‑browser. For weight matrices up to 4096×4096, a WebGL2 fragment shader approach (treating weights as texture data) can provide real‑time updates as compression parameters change. The implementation uses "GPU‑accelerated matrix operations on the GPU" via WebGL2 shaders.

**Color scale: d3‑scale‑chromatic**

Scientific color scales from d3 (`interpolateViridis`, `interpolatePlasma`, `interpolateRdYlBu`) are used to encode the impact magnitude. The `interpolateRdYlBu` scale (red=high impact, blue=low impact) is particularly intuitive for compression visualization.

#### 4.3 Component Hierarchy

```
CompressionImpactHeatmap
├── DeckGL (canvas)
│   ├── ImpactHeatmapLayer (custom deck.gl CompositeLayer)
│   │   ├── BitmapLayer (renders weight matrix as 2D image)
│   │   ├── ColumnLayer (renders singular value bars)
│   │   └── ScatterplotLayer (highlights top‑changed entries)
│   └── TextLayer (axis labels: input features, output features)
├── ImpactTooltip (HTML overlay, positioned via deck.gl picking)
│   ├── OriginalValue
│   ├── CompressedValue
│   ├── DeltaIndicator
│   └── ParameterLocation
├── LayerSelector (dropdown: which weight matrix to display)
│   └── Shows all weight matrices in the selected component
├── StrategySelector (dropdown: which compression strategy to preview)
│   └── Shows only strategies applicable to this component
└── ZoomControls
    ├── ZoomIn / ZoomOut buttons
    ├── FitToScreen button
    └── CoordinateDisplay (current cursor position in matrix)
```

#### 4.4 Data Flow

```
compressionStore (Zustand)
    │
    │ selectedCompressor, compressorParams, originalWeights, compressedWeights
    ▼
useBrowserSVD() hook OR backend API
    │
    │ Computes compressed weights based on selected strategy and params
    │ For SVD: runs numpy.linalg.svd in Pyodide or Web Worker
    │ For pruning: applies threshold mask
    │ For quantization: applies uniform quantization
    ▼
impactMatrix = |originalWeights - compressedWeights| / max(|originalWeights|)
    │
    ▼
ImpactHeatmapLayer receives impactMatrix as data prop
    │
    │ Renders via WebGL2 texture (BitmapLayer) with color mapping
    ▼
User hovers over cell → deck.gl picking → ImpactTooltip displays exact values
```

#### 4.5 Key Implementation Details

**Custom deck.gl CompositeLayer**

The `ImpactHeatmapLayer` extends `deck.gl.CompositeLayer` and composites three sub‑layers:

1. **BitmapLayer:** Renders the full weight matrix as a 2D image where each pixel's color encodes the impact magnitude. For a 768×768 matrix, this is a 768×768 pixel image rendered via WebGL texture. The bitmap is regenerated on the GPU whenever compression parameters change (via `shouldUpdateState`).

2. **ColumnLayer:** Renders the singular value spectrum as vertical bars along the bottom edge. Each bar height represents a singular value; colored bars are retained, gray bars are pruned. This directly visualizes the SVD truncation.

3. **ScatterplotLayer:** Highlights the top‑N most affected weight entries (largest delta) as glowing dots, making it easy to spot where compression has the greatest impact.

**Web Worker SVD Computation**

The `svd.worker.ts` file implements off‑main‑thread SVD computation:

```typescript
// Conceptual architecture (not code)
// Main thread:
const svdWorker = wrap(new Worker('./svd.worker.ts'));
const { U, S, Vt } = await svdWorker.computeSVD(weightMatrix, targetRank);

// Worker thread:
// Option A: Pyodide
self.importScripts('pyodide.js');
const pyodide = await loadPyodide();
await pyodide.loadPackage(['numpy', 'scipy']);
const result = pyodide.runPython(`
    import numpy as np
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vt_r = Vt[:rank, :]
    (U_r.tolist(), S_r.tolist(), Vt_r.tolist())
`);

// Option B: Pure JS (for small matrices)
// Uses numeric.js or ml-matrix for SVD
```

**Pyodide Integration**

Pyodide is loaded lazily (only when the user first opens the Compression Studio) via dynamic import. The Pyodide runtime (~6.5MB compressed) is cached by the browser's service worker. For SVD computation, Pyodide loads `numpy` and `scipy` packages on‑demand.

**Progressive rendering strategy:**
- Matrices < 256×256: render instantly via Pyodide in main thread (fast enough)
- Matrices 256×256 to 1024×1024: compute in Web Worker, show skeleton loading state
- Matrices > 1024×1024: fetch pre‑computed impact data from backend API, show progressive loading

---

### 5. Zone 3: Live Math Panel — The LaTeX Computation Engine

#### 5.1 Core Concept

The Live Math Panel renders the **mathematical formula** for the selected compression method in real‑time, showing the actual numerical values plugged into the equation. It also displays **parameter count before/after** and **estimated FLOPs reduction**. The panel is inspired by 3Blue1Brown's Manim animations and the SVD‑vis interactive tool.

#### 5.2 Technology Choices

**LaTeX rendering: KaTeX**

KaTeX is chosen over MathJax for its speed (renders synchronously, no DOM scanning) and smaller bundle size. KaTeX renders LaTeX strings to HTML in under 5ms for typical formulas, which is critical for real‑time updates as the user drags sliders.

**Numerical computation: Pyodide (NumPy) or pure JS**

Parameter counts and FLOPs estimations are computed client‑side:
- Parameter count: simple multiplication of matrix dimensions × (1 − compression_ratio)
- FLOPs estimation: uses formulas from the compression library (e.g., for SVD: `r(M+N)` vs original `MN`)
- Singular value energy: `Σ σ²_kept / Σ σ²_total` computed via NumPy in Pyodide

#### 5.3 Component Hierarchy

```
LiveMathPanel
├── CompressionFormulaDisplay
│   ├── KaTeX rendered formula (e.g., "W ≈ U_r Σ_r V_r^T")
│   ├── VariableHighlighter (highlights variables as user hovers visualization)
│   └── NumericalSubstitution (shows actual numbers: "W ≈ U_32 · Σ_32 · V_32^T")
├── ParameterCountComparison
│   ├── BeforeBar (horizontal bar, full width)
│   └── AfterBar (overlay, reduced width, animated transition)
├── FLOPsEstimator
│   ├── OriginalFLOPs (number with label)
│   ├── CompressedFLOPs (number with label)
│   └── SpeedupRatio ("4.2× speedup")
├── EnergyRetentionIndicator
│   ├── CircularGauge (0-100%, shows retained singular value energy)
│   └── EnergyLabel ("99.7% of energy retained")
└── BudgetTracker (if budget constraint is set)
    ├── CurrentSize ("187MB / 200MB")
    ├── ProgressBar (colored bar, green→yellow→red as budget fills)
    └── OverBudgetWarning (appears if compressed size exceeds budget)
```

#### 5.4 Key Implementation Details

**Real‑time formula updates**

The formula display uses a debounced (16ms) update loop: as the user drags a rank slider, the KaTeX string is regenerated with the new numerical values and re‑rendered. KaTeX's synchronous rendering ensures no flicker.

**Supported compression formula templates:**

Each compression method in the registry has an associated formula template:

```typescript
const FORMULA_TEMPLATES: Record<string, FormulaTemplate> = {
  'a3': {
    latex: 'W \\approx U_{{$rank}} \\Sigma_{{$rank}} V_{{$rank}}^T',
    variables: ['rank'],
    paramCountFn: (m, n, rank) => rank * (m + n),
    flopsFn: (m, n, rank) => rank * (m + n),
  },
  'ast_pruner': {
    latex: 'W_{2:4} = \\text{TopK}_{k=2}(W_{ij}), \\; j \\in \\{0,4,8,...\\}',
    variables: [],
    paramCountFn: (m, n) => m * n * 0.5,  // 50% sparsity
    flopsFn: (m, n) => m * n * 0.5,        // 50% fewer multiplies
  },
  'kmeans': {
    latex: 'W \\approx \\text{KMeans}_{k={$clusters}}(W)',
    variables: ['clusters'],
    paramCountFn: (m, n, clusters) => clusters + m * n * Math.ceil(Math.log2(clusters)) / 32,
    flopsFn: (m, n) => m * n,  // No FLOP reduction
  },
  // ... templates for all 60+ compressors
};
```

**Parameter count bar chart animation**

The before/after parameter count comparison uses an animated bar chart (Framer Motion `layout` animation). When the user changes compression parameters, the "after" bar smoothly animates to its new width, with the percentage reduction displayed as a label that counts up/down.

---

### 6. Zone 4: Controls — The Compression Parameter Orchestrator

#### 6.1 Core Concept

The Controls Zone is the **command center** where users select compression strategies, adjust parameters via sliders, set budget constraints, and trigger compression. It is designed to be immediately understandable while exposing the full power of the 60+ compression library.

#### 6.2 Technology Choices

**Slider library: Custom React components + @radix‑ui/react‑slider**

Custom slider components are built on top of Radix UI's accessible slider primitive. Each slider is annotated with:
- Real‑time numerical value display
- A mini sparkline showing the parameter's effect on model size
- Color‑coded range indicators (green=recommended, yellow=aggressive, red=extreme)

**Compressor selector: Combobox with search**

A command palette‑style combobox (built with cmdk or @radix‑ui/react‑popover) allows users to search and filter through all 60+ compression methods. Each result shows:
- Compressor name
- Category badge (SVD, Pruning, Quantization, Hybrid)
- Short description
- Typical compression ratio range

**Budget constraint solver: Custom algorithm**

The budget constraint input accepts a target model size (in MB) and uses a **greedy allocation algorithm** to suggest per‑layer compression parameters that meet the budget. The solver runs in a Web Worker to avoid blocking the UI.

#### 6.3 Component Hierarchy

```
ParameterSliders
├── CompressorSelector (combobox with search)
│   └── CompressorOption (icon + name + category + description)
├── RankSlider (when compressor uses rank parameter)
│   ├── SliderTrack (with gradient fill)
│   ├── SliderThumb (draggable)
│   ├── RankValueLabel ("rank=64")
│   └── RankSparkline (mini chart showing parameter count vs rank)
├── SparsitySlider (when compressor uses sparsity)
│   ├── SliderTrack
│   ├── SparsityValueLabel ("sparsity=0.5")
│   └── SparsitySparkline
├── BitsSlider (when compressor uses quantization)
│   ├── DiscreteMarkers (ticks at 2, 4, 8, 16, 32)
│   ├── BitsValueLabel ("4-bit")
│   └── MemorySavingsLabel ("8× memory reduction")
├── BudgetConstraintInput
│   ├── TextInput ("200 MB")
│   ├── AutoAllocateButton ("Auto‑allocate compression budget")
│   └── BudgetFeasibilityIndicator (checkmark or warning)
├── CompressionPresetSelector
│   ├── PresetButton ("Fast" — speed‑optimized)
│   ├── PresetButton ("Accurate" — quality‑preserving)
│   └── PresetButton ("Balanced" — default compromise)
└── ApplyButton
    ├── PrimaryButton ("Compress Now")
    ├── ProgressIndicator (spinner during compression)
    └── SuccessCheckmark (on completion)
```

#### 6.4 Key Implementation Details

**Budget constraint solver algorithm**

```
Input: target_size_mb, model_layers (with parameter counts and compressibility scores)
Output: per_layer_params (rank, sparsity, bits for each layer)

Algorithm:
1. Calculate current total size: Σ layer_params_count × 4 bytes / (1024²)
2. Calculate required reduction: current_total - target_size_mb
3. Sort layers by compressibility_score (high score = can be compressed more)
4. For each layer (in descending compressibility order):
   a. Apply maximum compression (lowest rank, highest sparsity)
   b. Recalculate total size
   c. If total ≤ target_size_mb, stop and return params
5. If still over budget, show warning: "Target size not achievable with this compressor"
```

This solver runs in a Web Worker and returns results in < 100ms for models with up to 100 layers.

**Slider‑to‑visualization reactivity**

All sliders are connected to the Zustand store via `useStore` selectors. When a slider value changes:
1. Zustand store updates immediately (synchronous)
2. The Live Math Panel re‑renders (KaTeX, sub‑millisecond)
3. The Compression Impact Heatmap schedules a re‑computation (debounced at 100ms)
4. The Model Architecture Tree re‑renders the compressed view (debounced at 100ms)

This architecture ensures smooth slider dragging (60fps) without expensive recomputation on every frame.

---

### 7. Zone 5: Cross‑Platform Integration — The Action Pipeline

#### 7.1 Core Concept

The Cross‑Platform Integration Zone connects the Compression Studio to the rest of the NeuralOps platform — the compression pipeline, the leaderboard, the experiment launcher, and the model versioning system. A single "Compress Now" button triggers the full workflow.

#### 7.2 Technology Choices

**API calls: TanStack Query mutations**

TanStack Query's `useMutation` hook is used for the compression API call. It provides:
- Optimistic updates (immediately show "compressing" state)
- Automatic retry on failure (3 retries with exponential backoff)
- Cache invalidation (refresh model version list after compression)
- Progress tracking via `onMutate`, `onSuccess`, `onError` callbacks

**Real‑time progress: WebSocket**

During compression, the Django backend sends progress updates via WebSocket (e.g., "Compressing layer 5/12... 42%"). The `useCompressionProgress` hook subscribes to the WebSocket channel and updates the progress bar.

**Before/After Leaderboard: Your existing leaderboard infrastructure**

The compressed model can be submitted to the benchmark leaderboard via your existing `gpu_benchmark` app. The submission includes:
- Original model name and version
- Compression method and parameters
- Before/after metrics (accuracy, FLOPs, model size)
- The attention pattern visualization captured at the time of compression

#### 7.3 Component Hierarchy

```
OneClickApply
├── ApplyButton ("Compress Now")
│   └── onClick → triggers compression pipeline
├── CompressionProgressBar
│   ├── ProgressBar (animated, shows current layer and percentage)
│   ├── CurrentOperationLabel ("Compressing Layer 5: Q projection...")
│   └── ETAIndicator ("~2 minutes remaining")
└── CompressionResultCard (appears on completion)
    ├── SuccessMessage ("Model compressed successfully!")
    ├── StatsComparison
    │   ├── SizeBefore ("589 MB → 187 MB (68% reduction)")
    │   ├── FLOPsBefore ("2.3M → 0.4M (83% reduction)")
    │   └── AccuracyBefore (if benchmarked)
    ├── ActionButtons
    │   ├── "Download Model"
    │   ├── "Submit to Leaderboard"
    │   ├── "Start Experiment with Compressed Model"
    │   └── "Create Note with Compression Report"
    └── CompressionReportPreview (auto‑generated note)

BeforeAfterLeaderboard
├── LeaderboardTable (sortable by compression ratio, accuracy retention, etc.)
├── SubmitButton (opens submission modal)
└── SubmissionModal
    ├── CompressionSummary (auto‑filled)
    ├── OptionalNotes (text area)
    └── SubmitConfirmButton

CompressionHistoryPanel
├── HistoryTable (list of past compression runs for this model)
│   ├── DateColumn
│   ├── CompressorColumn
│   ├── ParamsColumn
│   ├── SizeReductionColumn
│   └── ActionsColumn (download, compare, delete)
└── CompareButton (select two runs for side‑by‑side comparison)
```

#### 7.4 Key Implementation Details

**Compression Pipeline Integration**

The `OneClickApply` component sends a POST request to your existing compression API:

```
POST /api/runs/start-compress/
Body: {
  "model_name": "bert-base-uncased",
  "compressor": "a3",
  "compressor_config": { "rank": 32 },
  "attention_type": "galerkin",
  "sample_text": "The quick brown fox jumps over the lazy dog",
  "save_to_registry": true,
  "submit_to_leaderboard": true
}
```

The backend processes the compression (using your existing 60+ compression library), saves the compressed model to the model versioning system, and returns a run ID. The frontend then subscribes to WebSocket updates for progress tracking.

**Auto‑generated compression report**

After compression completes, an auto‑generated note is created using your `notes/report_generator.py`. The note includes:
- The compression method and parameters
- Before/after parameter counts and FLOPs estimates
- The singular value spectrum (embedded as SVG)
- The compression impact heatmap (embedded as PNG)
- Links to the compressed model version and leaderboard submission

**Attention pattern capture**

At the moment of compression, the attention pattern visualization is serialized (including layer/head selection, sample text, and the rendered SVG). This is stored as metadata on the compression run and linked to both the model version and the compression report.

---

## 8. State Management Architecture

### 8.1 Dual‑Store Architecture: Zustand + XState

The Compression Studio uses a **dual‑store** approach:

**Zustand** manages the **application data state** — the model being compressed, the selected compressor, the slider values, the compressed architecture preview, and the impact matrix. Zustand is chosen for its simplicity, performance (no provider wrapper needed), and excellent TypeScript support. The Workflow Designer UI for Giselle demonstrates this pattern: React Flow for graph visualization combined with Zustand for state management.

**XState** manages the **compression workflow state machine** — the sequence of states the compression process goes through. XState provides formal state machine semantics: each state has defined transitions, guards, and actions. This prevents "impossible states" (e.g., "comparing" before "compressing") and makes the workflow predictable and testable. XState is recommended for "complex orchestration with guards, timers, and hierarchies".

### 8.2 XState Machine Definition

```
States:
  idle
    → analyzing (on SELECT_COMPRESSOR)
  analyzing
    → ready (on ANALYSIS_COMPLETE)
    → error (on ANALYSIS_FAILED)
  ready
    → compressing (on APPLY_COMPRESSION)
    → idle (on CLEAR_SELECTION)
  compressing
    → completed (on COMPRESSION_SUCCESS)
    → failed (on COMPRESSION_FAILED)
    → idle (on CANCEL)
  completed
    → comparing (on COMPARE_WITH_ORIGINAL)
    → idle (on NEW_COMPRESSION)
    → submitted (on SUBMIT_TO_LEADERBOARD)
  comparing
    → completed (on CLOSE_COMPARISON)
    → submitted (on SUBMIT_TO_LEADERBOARD)
  submitted
    → completed (on CLOSE)
    → idle (on NEW_COMPRESSION)

Guards:
  canCompress: selectedCompressor !== null && selectedModel !== null
  canSubmit: compressionRunId !== null
  hasBudget: budgetConstraint !== null

Actions (side effects):
  onCompressing: start WebSocket subscription, show progress bar
  onCompleted: fetch compressed model architecture, update Zustand store
  onFailed: show error toast, log to activity feed
```

### 8.3 Zustand Store Design

The Zustand store is sliced into logical domains using the `slice` pattern:

```typescript
// compressionStore.ts — conceptual structure
interface CompressionStore {
  // Model slice
  model: ModelArchitecture | null;
  setModel: (model: ModelArchitecture) => void;

  // Compressor slice
  selectedCompressor: string | null;
  compressorParams: Record<string, number>;
  setCompressor: (name: string) => void;
  setParam: (key: string, value: number) => void;

  // Preview slice
  compressedArchitecture: ModelArchitecture | null;
  impactMatrix: Float32Array | null;
  paramCountBefore: number;
  paramCountAfter: number;
  flopsBefore: number;
  flopsAfter: number;

  // Budget slice
  budgetConstraint: number | null; // in MB
  setBudget: (mb: number) => void;
  isBudgetFeasible: boolean;

  // UI slice
  selectedNode: string | null;
  splitViewEnabled: boolean;
  heatmapLayer: number | null;
  setSelectedNode: (nodeId: string | null) => void;
  toggleSplitView: () => void;

  // History slice
  compressionHistory: CompressionRun[];
  addToHistory: (run: CompressionRun) => void;
}
```

---

## 9. Performance & Rendering Strategy

### 9.1 Rendering Tiers

The Compression Studio implements a **three‑tier rendering strategy** based on the complexity of the visualization:

| Tier | Data Size | Rendering Approach | Target FPS |
|------|-----------|-------------------|------------|
| **Tier 1: Small** | < 10K data points (single weight matrix < 100×100) | Standard React rendering (SVG with React) | 60fps |
| **Tier 2: Medium** | 10K–500K data points (weight matrix 100×100 to 700×700) | deck.gl BitmapLayer via WebGL2 texture | 60fps |
| **Tier 3: Large** | > 500K data points (weight matrix > 700×700) | deck.gl with progressive loading + Web Worker computation | 30fps (acceptable for static exploration) |

### 9.2 Memory Management

**Float32Array for impact matrices:** The impact matrix is stored as a `Float32Array` (4 bytes per entry) rather than a regular JavaScript array (8+ bytes per entry), reducing memory usage by 50%+ for large matrices.

**Web Worker offloading:** SVD computation and compression simulation are offloaded to Web Workers. The main thread only receives the results (U, S, Vt matrices), keeping it responsive for UI interactions.

**Lazy Pyodide loading:** Pyodide is loaded only when the user first opens the Compression Studio. The 6.5MB download is cached by the browser's service worker. Subsequent visits load Pyodide from cache in < 1 second.

### 9.3 Debouncing Strategy

```
User drags slider
  │
  ├─→ Zustand store updates (immediate, synchronous)
  │
  ├─→ Live Math Panel re‑renders (immediate, KaTeX < 5ms)
  │
  ├─→ Parameter count / FLOPs recompute (immediate, simple arithmetic)
  │
  ├─→ Architecture Tree compressed preview (debounced 100ms)
  │
  └─→ Impact Heatmap recompute (debounced 200ms, Web Worker)
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1‑3)
- Set up Compression Studio project structure
- Implement Zustand + XState dual‑store architecture
- Build ModelArchitectureTree with React Flow
- Implement basic compressor selector and parameter sliders
- Integrate KaTeX for formula rendering
- Build parameter count / FLOPs estimation logic
- Connect to existing compression API

### Phase 2: WebGL Visualization (Weeks 4‑6)
- Integrate deck.gl and build ImpactHeatmapLayer
- Implement Web Worker SVD computation (Pyodide)
- Build Before/After Split View
- Implement SingularValueSpectrum chart
- Add budget constraint solver

### Phase 3: Cross‑Platform Integration (Weeks 7‑8)
- Build OneClickApply pipeline
- Integrate WebSocket progress tracking
- Build BeforeAfterLeaderboard submission flow
- Implement auto‑generated compression reports
- Build CompressionHistoryPanel

### Phase 4: Polish & Performance (Weeks 9‑10)
- Optimize Web Worker computation for large matrices
- Implement progressive loading for large heatmaps
- Add animation transitions (Framer Motion)
- Build NodeInspector detail panel
- Add compression presets
- Accessibility audit and fixes
- E2E testing with Playwright

---

## Summary

The Compression Studio is a **six‑layer, multi‑runtime application** that combines:

- **React 18 + TypeScript** for the UI framework
- **React Flow** for the interactive model architecture tree
- **deck.gl + WebGL2** for GPU‑accelerated impact heatmaps
- **Pyodide (CPython 3.12 in WASM)** for client‑side SVD computation
- **Web Workers** for off‑main‑thread computation
- **KaTeX** for real‑time LaTeX formula rendering
- **Zustand + XState** for dual‑layer state management
- **TanStack Query** for server state caching
- **Framer Motion** for animation
- **WebSocket** for real‑time compression progress
- **Your existing 60+ compression library** for actual model compression
- **Your existing leaderboard, experiment launcher, notes, and model versioning apps** for cross‑platform integration

Every component is designed to be independently testable, lazy‑loaded for code splitting, and accessible via keyboard navigation and screen readers. The entire system operates at 60fps for interactive exploration while supporting matrices with millions of entries via GPU‑accelerated rendering.