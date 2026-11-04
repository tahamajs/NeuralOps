# 🧠 Full Pipeline Studio — Complete Implementation Architecture

## "From Input Token to Final Prediction — Every Step Visible, Every Value Inspectable"

---

## TABLE OF CONTENTS

1. [Overall Architecture & Technology Stack](#1-overall-architecture--technology-stack)
2. [Model Execution Engine](#2-model-execution-engine)
3. [Data Flow & State Management](#3-data-flow--state-management)
4. [Component Architecture & Hierarchy](#4-component-architecture--hierarchy)
5. [Zone-by-Zone Implementation Details](#5-zone-by-zone-implementation-details)
6. [Animation & Visual Effects Strategy](#6-animation--visual-effects-strategy)
7. [Performance Optimization](#7-performance-optimization)
8. [Integration With Existing Platform](#8-integration-with-existing-platform)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Overall Architecture & Technology Stack

### 1.1 Three-Tier Architecture

The Full Pipeline Studio uses a **three-tier architecture** that separates model execution, data processing, and visualization rendering:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIER 3: VISUALIZATION (Browser)               │
│  React 18 + TypeScript + React Three Fiber + D3.js + GSAP      │
│  WebGL/WebGPU Canvas | DOM Overlays | CSS Animations            │
├─────────────────────────────────────────────────────────────────┤
│                    TIER 2: DATA PROCESSING (Web Worker)          │
│  Comlink for RPC | SharedArrayBuffer for zero-copy              │
│  Tensor serialization/deserialization | State aggregation       │
├─────────────────────────────────────────────────────────────────┤
│                    TIER 1: MODEL EXECUTION (Browser/Server)      │
│  Option A: Transformers.js + ONNX Runtime Web (WebGPU/WASM)     │
│  Option B: Pyodide + PyTorch (pure Python in browser)           │
│  Option C: Django Backend API (for large models)                │
└─────────────────────────────────────────────────────────────────┘
```

The Transformer Explainer project (Polo Club, Georgia Tech) pioneered this client‑side approach: "the frontend uses Svelte and D3 for interactive visualizations, while the backend uses ONNX runtime and HuggingFace's Transformers library to run the GPT-2 model in the browser". AnimatedLLM (2025) similarly "runs entirely in the browser, using pre‑computed traces of open LLMs applied on manually curated inputs". Your platform extends this by supporting **three execution backends** depending on model size and user context.

### 1.2 Technology Stack Detail

| Layer | Technology | Purpose | Why This Choice |
|-------|-----------|---------|-----------------|
| **UI Framework** | React 18 + TypeScript 5.x | Component architecture, state management, routing | Already your frontend stack; component model supports complex interactive UIs |
| **3D Rendering** | React Three Fiber (R3F) + Three.js 160+ | GPU-accelerated 3D visualization of the full pipeline | R3F "uses React Three Fiber with native Three.js for GPU-accelerated rendering, supports both WebGPU and WebGL backends" |
| **2D Visualizations** | D3.js v7 + d3-sankey | Sankey flow diagrams, attention heatmaps, embedding scatter plots | Transformer Explainer "adopts the Sankey diagram visual design, inspired by recent work viewing Transformers as dynamical systems, emphasizing how input data flows through model components" |
| **Animation Engine** | GSAP (GreenSock) + ScrollTrigger | Smooth timeline animations, scroll-driven reveals, step-by-step forward pass | GSAP can "sync WebGL shaders with HTML layout" and "transform static scenes into fluid, story-like sequences" |
| **Model Execution (Browser)** | Transformers.js + ONNX Runtime Web | Run GPT-2, BERT, DistilBERT directly in browser | "You can easily convert your pretrained PyTorch, TensorFlow, or JAX models to ONNX using Optimum" |
| **Model Execution (Python)** | Pyodide (CPython 3.11 compiled to WASM) | Run PyTorch hooks and custom analysis in browser | "Pyodide enables scientific computing, data visualization, and interactive notebooks—all in client-side web apps" |
| **Web Workers** | Comlink + SharedArrayBuffer | Offload heavy computation, zero-copy data transfer | Prevents UI thread blocking during model inference |
| **State Management** | Zustand + Immer | Immutable state updates for model activations across layers | Already in your stack; supports middleware for undo/redo |
| **Data Fetching** | TanStack Query (React Query) | Cache management for model outputs, pre-computed traces | Already in your stack |
| **Styling** | Tailwind CSS 3.x | Consistent design system | Already in your stack |
| **Math Rendering** | KaTeX (browser) or MathJax | Live LaTeX equation rendering in the Math Panel | Faster than MathJax for real-time updates |

---

## 2. Model Execution Engine

### 2.1 Three Execution Backends (Auto-Selected by Model Size)

The system must support three backends, automatically selected based on model size and user's hardware capabilities:

#### Backend A: Transformers.js + ONNX Runtime Web (Small Models: GPT-2, DistilBERT, BERT-tiny)

```
Model Size: < 500M parameters
Latency: 50-500ms per forward pass
Hardware: CPU (WASM SIMD) or GPU (WebGPU)
```

**How it works:**
1. Model is converted from PyTorch to ONNX format using HuggingFace Optimum: `optimum-cli export onnx --model bert-base-uncased bert_onnx/`
2. ONNX model files (.onnx) are loaded by Transformers.js in the browser
3. Transformers.js selects the best execution provider: WebGPU → WASM SIMD → WASM baseline
4. Intermediate activations are extracted using **ONNX intermediate outputs**: the model is split at each layer boundary, with intermediate tensors exposed as named outputs. This is achieved by modifying the ONNX graph to add output nodes after each transformer block during the export process.

**Implementation detail for activation extraction:**
```python
# Server-side: Export model with intermediate outputs
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained(
    "bert-base-uncased", export=True,
    # Custom config to output all hidden states
    output_hidden_states=True
)
# This produces an ONNX model where each layer's hidden states
# are available as separate output tensors
```

#### Backend B: Pyodide + PyTorch (Medium Models, Research Use)

```
Model Size: 500M - 2B parameters
Latency: 1-5 seconds per forward pass
Hardware: CPU only (WASM)
```

**How it works:**
1. Pyodide loads a CPython 3.11 runtime compiled to WebAssembly in the browser
2. A minimal PyTorch package (torch-only, no CUDA) is installed in the Pyodide environment
3. The model is loaded from HuggingFace Hub or from a pre-downloaded checkpoint
4. `torch.nn.Module.register_forward_hook()` is used to capture all intermediate activations exactly as in native Python — "Hook函数是PyTorch提供的一种强大工具，它允许我们在不修改模型主体结构的情况下，拦截并处理模型运行过程中的中间数据" (PyTorch hooks allow intercepting intermediate data without modifying the model structure)
5. Captured tensors are serialized and sent to the main thread via Comlink

#### Backend C: Django Backend API (Large Models, Production Use)

```
Model Size: > 2B parameters
Latency: 5-30 seconds
Hardware: Server GPU (NVIDIA A100/H100)
```

**How it works:**
1. The user's input text is sent to your existing Django API endpoint
2. The server loads the model using HuggingFace Transformers with `output_hidden_states=True` and `output_attentions=True`
3. All intermediate activations, attention weights, and hidden states are captured server-side
4. Results are streamed back via WebSocket (using your existing Channels infrastructure) as they become available — layer by layer
5. The frontend progressively renders each layer as it arrives, providing a real-time "streaming" experience

### 2.2 Activation Capture Protocol

Regardless of backend, the data structure returned to the visualization layer is standardized:

```typescript
interface ForwardPassTrace {
  model_name: string;
  input_text: string;
  tokens: string[];                    // Tokenized input
  token_ids: number[];                 // Token IDs
  
  embedding: {
    token_embeddings: number[][];      // [seq_len, d_model]
    position_embeddings: number[][];   // [seq_len, d_model]
    combined_embeddings: number[][];   // [seq_len, d_model]
  };
  
  layers: LayerTrace[];
  
  output: {
    logits: number[][];                // [seq_len, vocab_size]
    predictions: {                     // Top-k predictions per position
      token_id: number;
      token: string;
      probability: number;
    }[][];
  };
}

interface LayerTrace {
  layer_index: number;
  
  attention: {
    q_proj: number[][];               // Query projection weights
    k_proj: number[][];               // Key projection weights
    v_proj: number[][];               // Value projection weights
    heads: {
      head_index: number;
      attention_weights: number[][];  // [seq_len, seq_len]
      q_vectors: number[][];          // Query vectors [seq_len, head_dim]
      k_vectors: number[][];          // Key vectors
      v_vectors: number[][];          // Value vectors
      output: number[][];             // Head output [seq_len, head_dim]
    }[];
    combined_output: number[][];      // After concatenation + projection
  };
  
  ffn: {
    pre_activation: number[][];       // Before activation function
    post_activation: number[][];      // After GELU/ReLU
    output: number[][];               // After second linear layer
  };
  
  residual_connections: {
    after_attention: number[][];      // Residual stream after attention
    after_ffn: number[][];            // Residual stream after FFN
  };
  
  layer_norm: {
    attention_input_norm: number[][]; // After pre-attention LayerNorm
    ffn_input_norm: number[][];       // After pre-FFN LayerNorm
  };
}
```

### 2.3 Web Worker Architecture

To prevent UI freezing during model inference, **all model execution runs in a Web Worker**:

```
Main Thread (React)                    Web Worker (Model Execution)
─────────────────────                   ─────────────────────────────
Zustand Store ←── Comlink ──→          Model Runner
    │                                    │
    ├── Canvas Rendering (R3F)          ├── Transformers.js / Pyodide
    ├── DOM Updates (React)             ├── Hook Capture
    └── User Input Handling             └── Tensor Processing
```

Comlink is used for RPC (Remote Procedure Call) between the main thread and worker, making the communication feel like direct function calls while actually going through `postMessage`:

```typescript
// Main thread
import { wrap } from 'comlink';
const ModelRunner = wrap<typeof import('./model-runner.worker')>(
  new Worker(new URL('./model-runner.worker.ts', import.meta.url))
);

// This looks like a direct call but runs in the worker
const trace = await ModelRunner.runForwardPass('bert-base-uncased', inputText);
```

For large tensors (e.g., attention matrices), `SharedArrayBuffer` enables **zero-copy** data transfer between the worker and main thread, critical for real-time visualization performance — as demonstrated by VisionFlow: "implements real-time position streaming via SharedArrayBuffer with zero-copy reads".

---

## 3. Data Flow & State Management

### 3.1 Global State Architecture (Zustand + Immer)

```typescript
interface PipelineStudioState {
  // Model & Input
  modelId: string;
  inputText: string;
  executionBackend: 'transformers.js' | 'pyodide' | 'django-api';
  
  // Execution
  isRunning: boolean;
  currentStep: number;         // Which step in the forward pass (0 to totalSteps)
  totalSteps: number;
  playbackMode: 'manual' | 'auto';
  playbackSpeed: number;       // 0.25x to 4x
  
  // Data
  trace: ForwardPassTrace | null;
  
  // Visualization
  activeView: 'pipeline' | 'attention' | 'math' | 'analytics';
  selectedLayer: number;
  selectedHead: number;
  hoveredToken: number | null;
  hoveredComponent: string | null;  // e.g., 'layer_3.attention.head_7'
  
  // Math Panel
  showMathSteps: boolean;
  currentMathStep: number;
  
  // Pipeline Challenge
  challengeActive: boolean;
  challengeReconstruction: Record<string, number[][]>;  // User's reconstructed activations
  challengeScore: number | null;
}
```

### 3.2 Progressive Data Loading

For large models (Backend C), data arrives **progressively via WebSocket**, and the visualization updates layer by layer as data streams in. This is the same WebSocket infrastructure you already use for experiment monitoring:

```
WebSocket Message Flow:
──────────────────────────────────────────────────────────────────
Server → { type: "layer_start", layer_index: 0, data: LayerTrace }
Server → { type: "layer_complete", layer_index: 0 }
Server → { type: "layer_start", layer_index: 1, data: LayerTrace }
...
Server → { type: "forward_pass_complete", output: OutputTrace }
```

Each incoming layer is immediately merged into the Zustand store and rendered. The GSAP timeline automatically adjusts to accommodate new data as it arrives.

### 3.3 Undo/Redo for Exploration

Since the user can modify input text, change models, or toggle components, the state supports **undo/redo** via Zustand's temporal middleware:

```typescript
import { temporal } from 'zundo';

const usePipelineStore = create<SliceState>()(
  temporal(
    (set, get) => ({ /* ... state and actions ... */ }),
    { limit: 50 }  // Keep last 50 states
  )
);
```

This enables "go back and try a different input" or "undo that head ablation" workflows.

---

## 4. Component Architecture & Hierarchy

### 4.1 Full Component Tree

```
<PipelineStudio>                              ← Root: loads data, manages global state
├── <GlobalContextBar>                        ← Zone 1: Model selector, text input, breadcrumbs
│   ├── <ModelSelector />
│   ├── <TextInput />
│   ├── <BackendIndicator />                  ← Shows which execution backend is active
│   └── <BreadcrumbTrail />
│
├── <StudioLayout>                            ← Main 3-panel layout (resizable)
│   │
│   ├── <LeftPanel>                           ← Zone 4: Controls
│   │   ├── <PlaybackControls />              ← Play, pause, step forward/back, speed
│   │   ├── <LayerNavigator />                ← Clickable list of all layers
│   │   ├── <ComponentFilter />               ← Show/hide: Attention, FFN, Norm, Residual
│   │   └── <QuickActions />                  ← Train, Note, Report, Paper buttons
│   │
│   ├── <CenterPanel>                         ← Zone 2: Main Canvas
│   │   ├── <PipelineCanvas />                ← The big interactive pipeline visualization
│   │   │   ├── <TokenizationStage />
│   │   │   ├── <EmbeddingStage />
│   │   │   ├── <PositionalEncodingStage />
│   │   │   ├── <LayerBlock />                ← One per layer, rendered dynamically
│   │   │   │   ├── <LayerNormNode />
│   │   │   │   ├── <AttentionBlock />
│   │   │   │   │   ├── <QKVProjections />
│   │   │   │   │   ├── <AttentionHeatmapMini />
│   │   │   │   │   └── <OutputProjectionNode />
│   │   │   │   ├── <ResidualConnection />
│   │   │   │   └── <FFNBlock />
│   │   │   │       ├── <Linear1Node />
│   │   │   │       ├── <ActivationNode />
│   │   │   │       └── <Linear2Node />
│   │   │   ├── <OutputStage />
│   │   │   └── <PredictionStage />
│   │   ├── <SankeyOverlay />                 ← Optional Sankey flow overlay
│   │   └── <3DOverlay />                     ← Toggle to 3D R3F view
│   │
│   └── <RightPanel>                          ← Zone 3 & 5: Math + Analytics
│       ├── <MathPanel />
│       │   ├── <LiveEquation />
│       │   ├── <StepByStepAnimation />
│       │   └── <NumericalInspector />
│       └── <AnalyticsPanel />
│           ├── <LatencyBreakdown />
│           ├── <MemoryBreakdown />
│           ├── <ComponentLatencyChart />
│           └── <ActivationDistributionPlot />
│
├── <BottomBar>                               ← Zone 7: Social & Actions
│   ├── <ReactionBar />
│   ├── <DiscussionThread />
│   └── <ShareExportButtons />
│
└── <PipelineChallengeOverlay>                ← Zone 7: Gamified challenge mode
    ├── <ChallengePrompt />
    ├── <ReconstructionCanvas />
    └── <LeaderboardWidget />
```

### 4.2 Key Design Decisions

**Why separate the canvas into stages rather than a single monolithic SVG?**
Each stage (Tokenization, Embedding, Layer 1...N, Output) is a separate React component that can be independently mounted, animated, and optimized. This follows the Transformer Explainer's modular approach: "The architecture consists of five primary subsystems that work together to provide an interactive educational experience". When a stage is not visible (scrolled past), it can be unmounted or simplified to save memory.

**Why React Three Fiber for the 3D view instead of raw Three.js?**
R3F provides a declarative React API for Three.js, allowing the 3D pipeline to share state (via Zustand) with the rest of the React application. When the user hovers over a token in the 2D attention heatmap, the corresponding 3D sphere can highlight simultaneously — because both components read from the same `hoveredToken` state. This bidirectional linking is significantly harder with imperative Three.js code.

---

## 5. Zone-by-Zone Implementation Details

### 5.1 Zone 1: Global Context Bar

**Purpose:** Persistent header showing what model, text, and backend are active.

**Implementation:**
- Renders at the top of the viewport, `position: sticky`
- Uses your existing `ModelPicker`, `AttentionPicker`, and text input components
- The backend indicator is a small badge: 🟢 "Running in browser (WebGPU)" or 🔵 "Connected to server (A100)"
- Breadcrumbs are clickable: `Pipeline Studio > BERT-base > Layer 3 > Attention Head 7`

### 5.2 Zone 2: Main Pipeline Canvas (The Core)

**Purpose:** A single, unified interactive canvas showing the entire forward pass as a flowing diagram.

**Implementation strategy — Three rendering layers:**

#### Layer 1: DOM-based layout (CSS Grid/Flexbox)
The pipeline structure itself — the boxes representing each component, their labels, and the flow arrows between them — is rendered as standard DOM elements. Each stage is a `div` positioned using CSS Grid with `grid-template-columns: repeat(auto-fill, minmax(200px, 1fr))` for horizontal layout. This approach was chosen because:
- DOM elements are inherently accessible (screen readers can parse them)
- CSS transitions and transforms are GPU-accelerated
- Text rendering in DOM is sharper than canvas text

#### Layer 2: SVG overlay (D3.js)
The flowing connections between components — curved arrows showing data flow — are rendered as an SVG overlay positioned absolutely on top of the DOM layout. D3.js generates the path data:

```typescript
// Sankey-style flow between pipeline stages
const linkGenerator = d3.sankeyLinkHorizontal();
// For each connection between two stages:
svg.selectAll('path')
  .data(connections)
  .join('path')
  .attr('d', linkGenerator)
  .attr('stroke-width', d => Math.max(1, d.value * 20))
  .attr('fill', 'none')
  .attr('stroke', d => colorScale(d.layerIndex))
  .attr('opacity', 0.7);
```

The Sankey approach follows Transformer Explainer's design: it "adopts the Sankey diagram visual design, inspired by recent work viewing Transformers as dynamical systems, emphasizing how input data flows through model components".

#### Layer 3: WebGL overlay (React Three Fiber, optional)
When the user toggles the 3D view, a full-screen R3F canvas replaces the DOM/SVG view. This canvas uses:

- `@react-three/drei` for camera controls (`OrbitControls`)
- Custom GLSL shaders for particle effects (attention flow visualization)
- Instanced rendering for hundreds of identical neuron nodes
- Post-processing effects (bloom, glow) via `@react-three/postprocessing`

The 3D view maps each token to a sphere in 3D space (via t-SNE/UMAP of hidden states), with attention connections as curved tubes between spheres. The entire pipeline is "flyable" — users can navigate through layers as if flying through a tunnel.

**Interaction Model:**

- **Click** any component (e.g., "Attention Head 3") → zooms into that component's detailed dashboard (Zone 5 updates)
- **Hover** over any component → highlights its connections, shows tooltip with key values
- **Hover** over a token → highlights all connections from/to that token across all layers
- **Scroll** (with GSAP ScrollTrigger) → advances through the forward pass step by step. As the user scrolls, the animation progresses from Tokenization → Embedding → Layer 1 → Layer 2 → ... → Prediction. This creates a "story-telling" experience where the user physically moves through the model.

**Implementation of scroll-driven animation:**

```typescript
// Using GSAP ScrollTrigger for the scroll-driven forward pass
useEffect(() => {
  const tl = gsap.timeline({
    scrollTrigger: {
      trigger: '.pipeline-canvas',
      start: 'top top',
      end: 'bottom bottom',
      scrub: 1,        // Smooth scrubbing, 1 second lag
      pin: true,       // Pin the canvas while scrolling
    }
  });
  
  // Each stage appears and animates as user scrolls
  tl.from('.tokenization-stage', { opacity: 0, y: 50 })
    .from('.embedding-stage', { opacity: 0, y: 50 })
    .from('.layer-0', { opacity: 0, scale: 0.8 })
    .from('.layer-1', { opacity: 0, scale: 0.8 })
    // ... and so on
    .from('.prediction-stage', { opacity: 0, y: -50 });
    
  return () => tl.kill();
}, [trace]);
```

GSAP with ScrollTrigger is chosen because it can "transform static scenes into fluid, story-like sequences" and "sync WebGL shaders with HTML layout". For scroll-driven animation, GSAP + ScrollTrigger is the industry standard — used by Apple, Google, and most award-winning interactive websites.

### 5.3 Zone 3: Live Math Panel

**Purpose:** Display the mathematical equations underlying each component in the pipeline, with animated step-by-step computation.

**Implementation:**

The Math Panel uses **KaTeX** for rendering LaTeX equations in real-time. KaTeX is chosen over MathJax because it renders synchronously (no flicker) and is about 5x faster for repeated updates.

The panel has three sub-sections:

#### 5.3.1 Live Equation Display

As the user hovers over different pipeline components, the equation updates to show the relevant formula:

```
Hover over "Attention" node:
─────────────────────────────────────────
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

Where:
  Q = X · W_Q    (query projection)
  K = X · W_K    (key projection)
  V = X · W_V    (value projection)
  d_k = 64       (head dimension)
```

When the user clicks the equation, it expands to show **numerical values**:

```
Attention(Q, K, V) = softmax(QK^T / √64) · V

With current values:
  Q[3] · K[7] = 0.847  →  ÷ 8  →  0.106  →  softmax  →  0.023
```

#### 5.3.2 Step-by-Step Computation Animation

When the user clicks "Show Steps," a GSAP timeline animates through each computation step:

1. **Matrix Multiplication**: Two rectangles (representing Q and K^T) slide together, and the resulting matrix appears
2. **Scaling**: A "÷ √d_k" label appears over each cell, values shrink proportionally
3. **Softmax**: Each row "lights up" as values are exponentiated (red glow for high values), then the row normalizes (all values sum to 1)
4. **Weighted Sum**: The attention weights are shown multiplying the V matrix, with weighted lines flowing to the output

This animation style is directly inspired by 3Blue1Brown's videos, which "built his own open source Python animation library, manim, to enable his YouTube work". While we use GSAP for browser performance rather than Manim (which renders video), the visual language is the same: "cinematic animation inspired by 3Blue1Brown, made with ManimCE".

#### 5.3.3 Numerical Inspector

A floating panel that follows the cursor, showing precise numerical values for whatever the user hovers over:

```
┌─────────────────────────────┐
│ Token: "fox"                │
│ Position: 3                 │
│                             │
│ Q · K (token "quick"):      │
│   Raw dot product: 0.847    │
│   Scaled (÷ √64):  0.106    │
│   After softmax:    0.023   │
│                             │
│ Contribution to output:     │
│   Weight: 0.023             │
│   Value vector norm: 1.24   │
│   Weighted norm: 0.029      │
└─────────────────────────────┘
```

### 5.4 Zone 4: Control Panel

**Purpose:** Layer navigation, playback controls, and quick actions.

**Implementation:**
- **Layer Navigator**: A vertical list of all layers, each with a miniature preview of its attention pattern (rendered as a tiny SVG heatmap, 30×30px). Click a layer to zoom into it. The active layer is highlighted with a glow effect.
- **Playback Controls**: A transport bar (like a video player) with play/pause, step forward, step backward, speed selector (0.25×, 0.5×, 1×, 2×, 4×), and a scrubber showing progress through the forward pass.
- **Component Filter**: Toggle checkboxes: ☑ Attention, ☑ FFN, ☑ LayerNorm, ☑ Residual. Unchecking a component fades it out in the pipeline visualization and removes it from the math panel.
- **Quick Actions**: Four buttons that connect to other platform apps — "⚡ Train with This" → experiment launcher, "📝 Create Note" → notes app, "📄 Generate Report" → paper app, "🔗 Copy Citation" → clipboard.

### 5.5 Zone 5: Analytics & Diagnostics

**Purpose:** Quantitative breakdown of latency, memory, and activation patterns.

**Implementation:**
- **Latency Breakdown Chart**: A horizontal stacked bar chart (using Recharts, already in your stack) showing: Embedding time | Layer 1 (Attention + FFN) | Layer 2 | ... | Output. Each segment is clickable to zoom into sub-components.
- **Memory Breakdown**: A treemap visualization (using D3.js treemap) showing memory allocation per component. Larger rectangles = more memory. Color intensity = memory utilization percentage.
- **Component Latency Comparison**: When comparing two attention types (e.g., Standard MHA vs. Galerkin), a side-by-side bar chart shows the latency difference for each component.
- **Activation Distribution Plot**: A violin plot or histogram showing the distribution of activation values in a selected layer, revealing saturation patterns (are neurons dying?) or unusual distributions.

### 5.6 Zone 7: Social & Pipeline Challenge

**Purpose:** Community interaction and a gamified challenge mode.

#### 5.6.1 Pipeline Challenge Mode

This is the core gamified feature. The challenge works as follows:

1. The system shows the user the **input text** and the **final output prediction** of a model
2. The intermediate activations (attention weights, hidden states) are **hidden** — the pipeline appears as a "mystery" with grayed-out components
3. The user must **reconstruct the intermediate activations** by dragging attention weights, adjusting hidden state values, or selecting from multiple-choice options
4. After submitting, the system reveals the actual activations and computes a **reconstruction score** based on cosine similarity, KL divergence, or exact match percentage
5. Scores are posted to a **community leaderboard** (integrated with your existing `leaderboards` app)

**Implementation of the reconstruction interface:**

Each grayed-out component becomes an interactive "puzzle piece":
- **Attention weights**: The user sees a 2D grid and can drag sliders on individual cells or use a "paint" tool to distribute attention across tokens
- **Hidden states**: The user sees a bar chart of activation magnitudes and can adjust them
- **FFN activations**: The user sees a simplified view of neuron activations (top-10 neurons by magnitude) and can toggle them on/off

The challenge system uses **Polis-style consensus voting** to aggregate community results: "Polis uses machine learning and statistical algorithms to analyze participants' voting behavior" and "visualizes the distribution of opinions" using PCA and k-means. Similarly, the Pipeline Challenge clusters users by their reconstruction strategies, revealing common patterns and misconceptions.

#### 5.6.2 Integration with Comparison Mode

The pipeline studio supports **side-by-side comparison** of two different models, inputs, or attention mechanisms. This is implemented as a split-screen view:

```
┌───────────────────────┐ ┌───────────────────────┐
│   Model A: BERT-base  │ │   Model B: Galerkin   │
│                       │ │                       │
│   [Pipeline Canvas]   │ │   [Pipeline Canvas]   │
│                       │ │                       │
│   Latency: 12.3ms     │ │   Latency: 4.7ms      │
│   Memory: 438MB       │ │   Memory: 112MB       │
└───────────────────────┘ └───────────────────────┘
```

Both canvases are linked — hovering over a component in one highlights the corresponding component in the other. The math panel shows a **diff view** of the equations.

---

## 6. Animation & Visual Effects Strategy

### 6.1 The Three Animation Systems

The pipeline studio uses three complementary animation systems, each for a specific purpose:

| Animation System | Technology | Use Case |
|-----------------|-----------|----------|
| **CSS Transitions** | Tailwind CSS + `framer-motion` | Layout changes, component fade-in/out, hover effects, responsive resizing |
| **GSAP Timeline** | GSAP + ScrollTrigger | Scroll-driven forward pass progression, step-by-step math animations, complex coordinated sequences |
| **WebGL Shaders** | React Three Fiber + GLSL | Particle flow effects, 3D token movement, real-time data visualization in 3D space |

### 6.2 Coordinated Animation Flow

The three animation systems are coordinated through the Zustand store:

```typescript
// The store acts as a "conductor" synchronizing all animation systems
const usePipelineStore = create((set, get) => ({
  currentStep: 0,
  
  advanceStep: () => {
    const next = get().currentStep + 1;
    set({ currentStep: next });
    
    // GSAP timeline advances
    gsapTimeline.seek(next / totalSteps);
    
    // CSS classes update reactively via React state
    // R3F frame updates via useFrame hook reading currentStep
  }
}));
```

### 6.3 Particle Flow Effect (GLSL Shader)

For the "flowing attention" effect, a custom GLSL fragment shader runs on the GPU:

```
// Fragment shader pseudocode for attention flow particles
// Runs thousands of particles in parallel on the GPU
// Each particle represents an "attention quantum" flowing from query to key

uniform float uTime;
uniform vec2 uQueryPositions[128];  // Positions of query tokens
uniform vec2 uKeyPositions[128];    // Positions of key tokens
uniform float uAttentionWeights[128][128];  // Attention matrix

void main() {
  // Each particle is assigned a (query, key) pair based on attention weight
  // It flows along a bezier curve from query position to key position
  // Opacity and speed are proportional to attention weight
  // Glow effect via gaussian blur in a post-processing pass
}
```

This approach leverages the GPU for massive parallelism — rendering 10,000+ particles at 60fps — which would be impossible with DOM or SVG alone. VisionFlow demonstrates this pattern: "implements real-time position streaming via SharedArrayBuffer with zero-copy reads" for GPU-based rendering.

### 6.4 Color Palette for Pipeline Stages

Each stage in the pipeline has a distinct color, creating a visual "rainbow" that helps users understand the flow:

```
Tokenization:     🔵 Blue (#3B82F6)
Embedding:        🟣 Purple (#8B5CF6)
Positional Enc:   🟢 Green (#10B981)
Attention:        🔴 Red (#EF4444)
FFN:              🟡 Yellow (#F59E0B)
LayerNorm:        ⚪ Gray (#6B7280)
Residual:         🟠 Orange (#F97316)
Output/Prediction:🟦 Cyan (#06B6D4)
```

---

## 7. Performance Optimization

### 7.1 Data Transfer Optimization

For large models, the activation trace can be tens of megabytes. Transferring this from worker to main thread must be optimized:

1. **Float32 → Float16 conversion**: Attention weights don't need 32-bit precision for visualization. Quantize to Float16 (via `Float16Array`) before transfer, halving the data size.
2. **Sparse representation for attention**: Most attention weights are near zero. Only transmit values above a threshold (e.g., > 0.01), with a sparse matrix format (CSR/CSC).
3. **Progressive loading**: Transmit layer by layer rather than the entire trace at once. The visualization can start rendering Layer 0 while Layer 11 is still being computed.

### 7.2 Rendering Optimization

1. **Virtualization**: For models with many layers (e.g., GPT-2 Large with 36 layers), only render layers currently visible in the viewport. Layers above and below are placeholder divs with fixed height.
2. **Canvas layering**: Use multiple canvas elements stacked via `z-index` rather than redrawing the entire canvas when only one element changes.
3. **Web Worker for D3 layout**: D3's Sankey layout computation can be CPU-intensive. Offload it to a Web Worker.
4. **Memoization**: All derived values (computed attention statistics, normalized scores) are memoized with `useMemo` to avoid recalculation on every render.

### 7.3 Memory Management

1. **LRU Cache for model traces**: Keep the last 5 forward pass traces in memory. When the user runs a 6th input, evict the oldest.
2. **IndexedDB persistence**: Optionally persist traces to IndexedDB for offline access and historical comparison.

---

## 8. Integration With Existing Platform

### 8.1 Connections to Your Django Apps

| Pipeline Studio Feature | Connected Platform App | How It Connects |
|------------------------|----------------------|-----------------|
| Model selector dropdown | `model_versioning` | Lists all registered model versions from your registry |
| "Train with This" button | `experiment` launcher | Opens launcher pre-filled with model, attention type, text |
| "Create Note" button | `notes` | Creates a ResearcherNote with embedded visualization SVGs |
| "Generate Report" button | `paper` | Opens LaTeX editor with pre-written Methods section |
| Pipeline Challenge | `leaderboards` | Challenge scores posted to leaderboard |
| Share/Discuss | `community` | Creates Q&A thread with embedded interactive visualization |
| Attention type selector | `ATTENTION_REGISTRY` | Lists all registered attention types |
| Model execution | GPU Marketplace (`gpu_marketplace`) | For backend C, jobs are dispatched to provider GPUs |
| Reputation/Badges | `gpu_reputation` | Earn badges for challenge wins, pattern discoveries |

### 8.2 API Endpoints (New)

These endpoints need to be added to your Django backend:

```
POST   /api/pipeline/run-forward-pass/       ← Run a forward pass and return trace
GET    /api/pipeline/traces/                  ← List saved traces
GET    /api/pipeline/traces/{id}/             ← Get a saved trace
POST   /api/pipeline/traces/{id}/share/       ← Share a trace with community
POST   /api/pipeline/challenges/submit/       ← Submit challenge reconstruction
GET    /api/pipeline/challenges/leaderboard/  ← Challenge leaderboard
```

---

## 9. Implementation Roadmap

### Phase 1: Core Pipeline Canvas (Weeks 1-3)

**Goal:** A working, scrollable pipeline visualization showing the full forward pass.

**Tasks:**
1. Set up the data flow: Web Worker + Comlink + Zustand store
2. Implement Backend A (Transformers.js + ONNX Runtime Web) for GPT-2/DistilBERT
3. Build the DOM-based pipeline layout with CSS Grid
4. Implement SVG overlay for data flow arrows (D3.js Sankey)
5. Add GSAP ScrollTrigger for scroll-driven forward pass animation
6. Build the Layer Navigator and Playback Controls

**Deliverable:** User can select a small model, type text, and watch the full forward pass animate step by step as they scroll.

### Phase 2: Math Panel + Analytics (Weeks 4-5)

**Goal:** Live equations, step-by-step math animations, and latency/memory analytics.

**Tasks:**
1. Integrate KaTeX for live equation rendering
2. Build the Step-by-Step Computation Animation (GSAP timeline)
3. Implement the Numerical Inspector floating panel
4. Build Latency Breakdown Chart (Recharts)
5. Build Memory Breakdown Treemap (D3.js)
6. Add Component Filter toggles

**Deliverable:** User can hover over any component to see its math, click to expand step-by-step computation, and view latency/memory analytics.

### Phase 3: 3D View + Advanced Visualization (Weeks 6-8)

**Goal:** Immersive 3D pipeline view and particle flow effects.

**Tasks:**
1. Set up React Three Fiber canvas with camera controls
2. Build the 3D token-to-token attention visualization (spheres + tubes)
3. Implement GLSL particle flow shader
4. Add layer "tunnel" navigation (fly through layers)
5. Synchronize 2D and 3D views (hover in 2D highlights in 3D)

**Deliverable:** User can toggle between 2D and 3D views, with smooth transitions and real-time synchronization.

### Phase 4: Pipeline Challenge + Social Features (Weeks 9-10)

**Goal:** Gamified reconstruction challenge and community sharing.

**Tasks:**
1. Build the Challenge Mode (hide activations, user reconstructs)
2. Implement reconstruction scoring (cosine similarity, KL divergence)
3. Integrate with `leaderboards` for challenge rankings
4. Add share/discuss buttons connected to `community`
5. Build the side-by-side comparison mode (split screen)

**Deliverable:** Users can participate in challenges, compare models side-by-side, and share their explorations.

### Phase 5: Backend B + C Integration (Weeks 11-12)

**Goal:** Support for medium and large models via Pyodide and Django API.

**Tasks:**
1. Set up Pyodide + PyTorch in browser for medium models
2. Build the Django API endpoint for large model inference
3. Implement WebSocket streaming for progressive layer-by-layer data loading
4. Auto-select backend based on model size and hardware capabilities
5. Add GPU marketplace integration (dispatch inference jobs to provider GPUs)

**Deliverable:** Full pipeline works for any model size, from GPT-2 in the browser to Llama-70B on server GPUs.

### Phase 6: Polish & Performance (Weeks 13-14)

**Goal:** Smooth 60fps performance, accessibility, and cross-browser compatibility.

**Tasks:**
1. Performance profiling and optimization (Chrome DevTools, React Profiler)
2. Implement Float16 quantization for data transfer
3. Add progressive loading for large traces
4. Accessibility audit (screen readers, keyboard navigation, color contrast)
5. Cross-browser testing (Chrome, Firefox, Safari, Edge)
6. Mobile responsive design (simplified view for small screens)

**Deliverable:** Production-ready pipeline studio with 60fps performance and full accessibility.

---

## Summary: Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FULL PIPELINE STUDIO — TECH STACK                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    PRESENTATION LAYER (Browser)                  │    │
│  │                                                                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │ React 18 │  │ R3F +    │  │ D3.js v7 │  │ GSAP +           │ │    │
│  │  │ + TS 5.x │  │ Three.js │  │ + d3-    │  │ ScrollTrigger     │ │    │
│  │  │          │  │ 160+     │  │ sankey   │  │                   │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │    │
│  │       │              │              │               │             │    │
│  │       └──────────────┴──────────────┴───────────────┘             │    │
│  │                          │                                         │    │
│  │               Zustand Store (Global State)                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                │                                         │
│  ┌─────────────────────────────┴───────────────────────────────────┐    │
│  │                    DATA LAYER (Web Worker)                       │    │
│  │                                                                   │    │
│  │  Comlink (RPC)  │  SharedArrayBuffer  │  Float16Array (Quant)    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                │                                         │
│  ┌─────────────────────────────┴───────────────────────────────────┐    │
│  │                  EXECUTION LAYER (Multi-Backend)                 │    │
│  │                                                                   │    │
│  │  ┌────────────────┐  ┌────────────┐  ┌──────────────────────┐   │    │
│  │  │ Backend A      │  │ Backend B  │  │ Backend C            │   │    │
│  │  │ Transformers.js│  │ Pyodide +  │  │ Django API + GPU     │   │    │
│  │  │ + ONNX Runtime │  │ PyTorch    │  │ Marketplace          │   │    │
│  │  │ Web (WebGPU/   │  │ (WASM)     │  │ (Server-side)        │   │    │
│  │  │ WASM)          │  │            │  │                      │   │    │
│  │  │                │  │            │  │                      │   │    │
│  │  │ Model: <500M   │  │ Model:     │  │ Model: >2B           │   │    │
│  │  │ Latency: 50ms  │  │ 500M-2B    │  │ Latency: 5-30s       │   │    │
│  │  └────────────────┘  └────────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              PLATFORM INTEGRATION (Django Backend)               │    │
│  │                                                                   │    │
│  │  experiment │ notes │ paper │ leaderboards │ community │ gpu_    │    │
│  │  launcher   │       │       │              │           │ marketplace│   │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key References (From Research)

1. **Transformer Explainer** (Polo Club, Georgia Tech, 2024): "The frontend uses Svelte and D3 for interactive visualizations, while the backend uses ONNX runtime and HuggingFace's Transformers library to run the GPT-2 model in the browser" — directly validates the browser-based execution approach.

2. **AnimatedLLM** (2025): "AnimatedLLM runs entirely in the browser, using pre-computed traces of open LLMs applied on manually curated inputs" — validates the pre-computed trace approach for smooth visualization.

3. **GPT2-WebGL** (2025): Real-time visualization components that "update in real-time as the model generates text, providing an interactive window into the model's reasoning process" — validates WebGL for real-time activation visualization.

4. **React Three Fiber + WebGPU** (2025): "R3F with native Three.js for GPU-accelerated rendering, supports both WebGPU and WebGL backends, and implements real-time position streaming via SharedArrayBuffer with zero-copy reads" — validates the R3F + WebGPU approach.

5. **Pyodide** (2025): "Python 3.11+ in the browser via WebAssembly" for "interactive environments such as JupyterLite" — validates the Pyodide backend for Python-based model analysis.

6. **Transformers.js + ONNX** (2025): "You can easily convert your pretrained PyTorch, TensorFlow, or JAX models to ONNX using Optimum" and run them "in the browser using onnxruntime-web" — validates the ONNX export pipeline.

7. **GSAP + ScrollTrigger** (2025): Can "transform static scenes into fluid, story-like sequences" with "shader-based depth" and "scroll-controlled showcases with moving cameras and animated typography" — validates GSAP for scroll-driven animation.

8. **Polis** (consensus tool): "Uses machine learning and statistical algorithms to analyze participants' voting behavior" and "visualizes the distribution of opinions using PCA and k-means" — validates the Pipeline Challenge consensus mechanism.