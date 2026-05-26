Here is a **fully detailed implementation blueprint** for the **Circuit Tracer — QK & OV Circuit Explorer**, covering technologies, architectures, and implementation strategies.

---

## 1. TECHNOLOGY STACK & JUSTIFICATION

### 1.1 Core Frameworks

| Layer | Technology | Why |
|-------|-----------|-----|
| **ML Backend (Circuit Extraction)** | `TransformerLens` + `NNsight` + `pyvene` | TransformerLens provides standardized HookPoints on every activation for QK/OV circuit extraction across architectures; NNsight preserves exact model behavior through HuggingFace access; pyvene enables interventions and causal abstraction |
| **Browser-Side ML** | Pyodide + WebAssembly | Enables running Python ML code directly in the browser—no server needed for small models; Mozilla's wasm-agents blueprint proves this is production-ready |
| **GPU Compute** | WebGPU (WGSL compute shaders) | Native GPU matrix operations in the browser—critical for real-time attention pattern computation on matrices up to 2048×2048 tokens |
| **3D Graph Visualization** | React Three Fiber (`@react-three/fiber`) + `r3f-forcegraph` | Declarative 3D graph rendering in React; r3f-forcegraph provides force-directed layouts with directional particles, node collision, expand/collapse |
| **2D/Canvas Graph** | `Cytoscape.js` + `Deck.gl GraphLayer` | Cytoscape for interactive 2D circuit graphs with built-in layouts; Deck.gl for GPU-accelerated rendering of large graphs via WebGL2/WebGPU |
| **State Management** | Zustand + Immer | Lightweight, performant state for circuit graph data, selection state, and intervention parameters |
| **Data Fetching** | React Query (`@tanstack/react-query`) | Caching, background refetching, and optimistic updates for circuit analysis results |

### 1.2 Key Dependencies

```
@react-three/fiber @react-three/drei three
r3f-forcegraph three-forcegraph
cytoscape cytoscape-dagre cytoscape-cose-bilkent
deck.gl @deck.gl/layers @deck.gl/react
zustand immer
@tanstack/react-query
d3-scale d3-color
```

---

## 2. CIRCUIT EXTRACTION ENGINE (Backend)

### 2.1 Architecture Overview

The circuit extraction pipeline follows the **Transformer Circuits mathematical framework** where every attention head is understood in terms of two matrices:

- **QK Circuit**: `Score(q,k) = (W_Q · x_i)^T · (W_K · x_j) / √d` — determines **where** the head attends
- **OV Circuit**: `Output = Σ_j A_ij · (W_V · x_j) · W_O` — determines **what information** the head reads and writes

The extraction runs in **four phases**:

#### Phase 1: Hook Point Registration
Using `TransformerLens`'s HookPoint system, attach hooks to every attention head's Q, K, V, O projection outputs across all layers. This captures the full QK and OV matrices for any given input text. HookPoints provide standardized access: `blocks.{layer}.attn.hook_q`, `blocks.{layer}.attn.hook_k`, `blocks.{layer}.attn.hook_v`, `blocks.{layer}.attn.hook_z`.

#### Phase 2: Attribution Graph Construction
Following the Anthropic attribution graph methodology, compute the **direct effect** (edge weight) between every pair of non-zero features. This uses **cross-layer transcoders** (trained via sparse autoencoders) to decompose the residual stream into interpretable features. The edge weight from feature A to feature B equals the dot product of the gradient of B's activation with respect to A, multiplied by A's activation value.

#### Phase 3: Circuit Pruning
Apply a **threshold pruning** strategy: remove all edges below a configurable threshold (default 0.01). Then apply **transitive reduction** to remove redundant indirect edges—if A→B and B→C exist with high weight, and A→C exists with low weight, A→C may be pruned since it's already explained by the path A→B→C.

#### Phase 4: Automatic Circuit Classification
Run automated head type detection using **detection patterns** (positional, previous-token, induction, copy, etc.) as implemented in TransformerLens's `head_detector` module. Each head is classified by comparing its attention pattern against known templates.

### 2.2 Browser-Side Computation Architecture

For **client-side circuit extraction** (no server needed for small models):

```
┌─────────────────────────────────────────────────────┐
│                  MAIN THREAD (React)                 │
│  • UI rendering, user interactions, state management│
│  • Renders circuit graph using Cytoscape/R3F        │
└─────────────────┬───────────────────────────────────┘
                  │ postMessage
┌─────────────────▼───────────────────────────────────┐
│           WEB WORKER 1: Model Inference              │
│  • Loads model via Pyodide + TransformerLens         │
│  • Runs forward pass with HookPoints                 │
│  • Extracts QK and OV matrices                       │
│  • Posts results back to main thread                 │
└─────────────────┬───────────────────────────────────┘
                  │ SharedArrayBuffer
┌─────────────────▼───────────────────────────────────┐
│        WEBGPU COMPUTE SHADER (WGSL)                  │
│  • Performs QK^T matrix multiplication on GPU        │
│  • Computes attention weights via softmax             │
│  • Processes matrices up to 2048×2048 in real-time   │
│  • Handles OV projection computations                │
└─────────────────────────────────────────────────────┘
```

The Web Worker handles model inference via Pyodide, while matrix operations are offloaded to WebGPU compute shaders written in WGSL. A `SharedArrayBuffer` enables zero-copy data transfer between the worker and GPU.

---

## 3. CIRCUIT GRAPH VISUALIZATION (Canvas)

### 3.1 Bipartite Circuit Graph Design

The core visualization is a **bipartite directed graph** with two node types:
- **Feature Nodes**: Represent individual transcoder features (e.g., "detects capital cities," "detects Texas-related text")
- **Input/Output Nodes**: Represent input tokens (left side) and output logits (right side)

Edges flow left-to-right:
- **QK edges** (blue): Show attention routing—which query features attend to which key features
- **OV edges** (orange): Show information writing—which features write to which output directions

### 3.2 Dual Rendering Pipeline (2D + 3D)

#### 2D View (Cytoscape.js)
- **Layout**: Dagre hierarchical layout for top-to-bottom or left-to-right circuit display
- **Edge bundling**: Hierarchical edge bundling (TP-Bundle approach) to reduce visual clutter in large circuits
- **Interaction**: Hover on edge → tooltip showing weight value; click on node → highlight all connected edges; drag nodes to reposition
- **Supernodes**: User-defined groups of features (e.g., "capital-related features") can be collapsed into a "supernode" for high-level circuit views

#### 3D View (React Three Fiber + r3f-forcegraph)
- **Force-directed 3D layout**: Nodes positioned in 3D space; edges rendered as directional particle flows with arrow indicators
- **Node encoding**: Sphere size = feature activation magnitude; color = feature type (QK circuit = blue, OV circuit = orange, Input = green, Output = purple)
- **Edge encoding**: Particle density = edge weight; particle speed = edge confidence; particle color = circuit type
- **Camera controls**: Orbit, pan, zoom; double-click on node to "fly to" it

### 3.3 GPU-Accelerated Edge Rendering (Deck.gl GraphLayer)

For large circuits (100+ nodes, 1000+ edges), Deck.gl's `GraphLayer` provides WebGL2/WebGPU-based rendering:
- Renders nodes as instanced circles and edges as instanced line segments
- GPU aggregation for collision detection and label placement
- Handles real-time interaction at 60 FPS for graphs with thousands of nodes

---

## 4. QK CIRCUIT EXPLORER (Sub-View 1)

### 4.1 Data Model

For each attention head in each layer:

```
QKCircuit {
  headId: "L5_H7",
  queryFeatures: [{ id, label, activationValue }],
  keyFeatures: [{ id, label, activationValue }],
  edges: [{ sourceId, targetId, weight, type: "qk" }],
  attentionPattern: number[][], // raw attention matrix
  headType: "induction" | "previous_token" | "copy" | "positional" | "semantic" | "unknown"
}
```

### 4.2 Computation Pipeline

1. **Extract QK matrix**: From `blocks.{layer}.attn.hook_q` and `blocks.{layer}.attn.hook_k`, compute `QK^T / √d_head`. Apply causal mask if the model is autoregressive.
2. **Feature decomposition**: Using pre-trained transcoders (sparse autoencoders), decompose both Q and K vectors into interpretable feature coefficients. Each transcoder feature has a human-readable label (e.g., "feature 27: detects 'is' relationship").
3. **Edge weight computation**: For feature pair (f_q, f_k), compute the dot product of their respective Q and K projection components, weighted by the feature activations.
4. **Pattern classification**: Run the TransformerLens `head_detector` to classify the head type based on its attention pattern template.

### 4.3 Visualization Layout

A **bipartite graph** rendered as two columns:
- **Left column**: Query features (labeled with activation values)
- **Right column**: Key features
- **Edges**: Arrows from query to key, color-coded by weight magnitude
- **Side panel**: Shows the raw attention matrix for this head as a heatmap

---

## 5. OV CIRCUIT EXPLORER (Sub-View 2)

### 5.1 Data Model

```
OVCircuit {
  headId: "L5_H7",
  valueFeatures: [{ id, label, activationValue }],
  outputFeatures: [{ id, label, activationValue }],
  edges: [{ sourceId, targetId, weight, type: "ov" }],
  headFunction: "copy" | "move" | "delete" | "transform" | "unknown"
}
```

### 5.2 Computation Pipeline

1. **Extract OV matrix**: From `blocks.{layer}.attn.hook_v` and `blocks.{layer}.attn.hook_z`, compute `W_V · x_j` and `W_O`. The OV circuit is: `Output = Σ_j A_ij · (W_V · x_j) · W_O`.
2. **Feature mapping**: Decompose V and O vectors into transcoder features.
3. **Functional classification**: Based on the OV matrix's eigenvalue spectrum: positive eigenvalues → copying function (amplifying features); negative eigenvalues → deleting function (suppressing features); mixed eigenvalues → transformation.

### 5.3 Visualization Layout

A **layered flow diagram**:
- **Bottom layer**: Value features (input to the OV circuit)
- **Middle layer**: The attention-weighted combination (visualized as a blend operation)
- **Top layer**: Output features (what the head writes to the residual stream)
- **Connection lines** show the strength of each feature's contribution

---

## 6. INDUCTION HEAD DETECTOR (Automated)

### 6.1 Detection Algorithm

The induction head detector follows the Transformer Circuits mathematical framework:

An induction head has:
1. **OV circuit**: Behaves like a copying head — maps features X to `say(X)` (outputs that would make the model predict X)
2. **QK circuit**: "If the query token is X, attend to a key token preceded by X"

**Detection Steps:**
1. Compute the attention pattern for every head
2. For each head, measure how much attention mass is on the "previous token" vs. "two tokens back"
3. Induction heads show a characteristic pattern: high attention to tokens that followed similar tokens
4. Validate by computing the OV circuit's eigenvalue spectrum — induction heads show positive eigenvalues in the copy-detection subspace

### 6.2 Implementation

Using TransformerLens's `head_detector` module with a custom detection pattern:

```
Detection Pattern (for autoregressive models):
- Expectation: attention[i, j] is highest when token_j == token_{i-1}
- This creates a "previous token matching" pattern
- Quantified by the "prefix matching score"
```

---

## 7. INTERVENTION PLAYGROUND

### 7.1 Causal Intervention Architecture

Following pyvene's intervention framework:

1. **Select source features** (e.g., feature 27 at token position 5)
2. **Set intervention value** (e.g., set activation to 0.0 to "knock out" the feature, or to 2.0 to amplify)
3. **Re-run forward pass** with the intervention applied at the HookPoint
4. **Measure effect**: Compare the output logits before and after intervention

### 7.2 Implementation

Using pyvene's `Intervention` API:
- Register `HookPoint` interventions at specific layers/heads
- Support counterfactual interventions (set to zero, set to specific value, swap with another position's activation)
- Visualize the effect as a "diff" in the circuit graph: edges that changed are highlighted in red (decreased) or green (increased)

---

## 8. CIRCUIT SHARING & ANNOTATION

### 8.1 Graph Serialization Format

Following the Anthropic Circuit Tracer's JSON format:

```
{
  "modelId": "gemma-2-2b",
  "prompt": "The capital of Texas is",
  "nodes": [
    { "id": "27_22605_10", "layer": 27, "featureIdx": 22605, "activation": 10.2 },
    { "id": "input_0", "type": "input", "token": "The" }
  ],
  "edges": [
    { "source": "input_5", "target": "27_22605_10", "weight": 0.83, "type": "qk" }
  ],
  "annotations": [
    { "nodeId": "27_22605_10", "label": "Detects state names" }
  ],
  "supernodes": [
    { "name": "Texas detection", "nodes": ["27_22605_10", "20_15589_10"] }
  ]
}
```

### 8.2 Neuronpedia Integration

Circuits can be uploaded to Neuronpedia for community exploration. The sharing workflow:
1. User clicks "Share Circuit" → generates a unique URL
2. The circuit is uploaded to the platform (or to Neuronpedia)
3. Other users can view, annotate, and fork the circuit
4. Annotations are collaborative (real-time via WebSockets)

---

## 9. PERFORMANCE OPTIMIZATION STRATEGY

### 9.1 Rendering Pipeline Tiers

| Graph Size | Technology | Latency Target |
|-----------|-----------|----------------|
| < 50 nodes | Cytoscape.js (SVG/Canvas) | < 16ms per frame |
| 50-500 nodes | Deck.gl GraphLayer (WebGL2) | < 16ms per frame |
| 500-5000 nodes | Helios-Web approach (WebGPU) | < 33ms per frame |
| > 5000 nodes | Level-of-detail + worker-based layout | < 100ms initial, 16ms interaction |

### 9.2 Computation Offloading Strategy

```
┌──────────────────────────────────────────────┐
│  QUICK OPERATIONS (< 50 nodes)               │
│  → Main thread (JavaScript)                  │
│  → Instantaneous response                    │
├──────────────────────────────────────────────┤
│  MEDIUM OPERATIONS (50-500 nodes)            │
│  → Web Worker for circuit extraction         │
│  → WebGPU for matrix operations              │
│  → < 100ms total latency                     │
├──────────────────────────────────────────────┤
│  LARGE OPERATIONS (500+ nodes)               │
│  → Server-side Python for circuit extraction │
│  → WebSocket streaming for results           │
│  → Progressive rendering as data arrives     │
└──────────────────────────────────────────────┘
```

---

## 10. INTEGRATION WITH EXISTING PLATFORM

The Circuit Tracer connects to your existing Django apps:

| Connection | Integration Point |
|-----------|------------------|
| **Models** | Load from `model_versioning` registry or HuggingFace Hub |
| **Experiments** | "Run Circuit Analysis" button on experiment detail page |
| **Notes** | Auto-generate note with circuit diagram + analysis |
| **Paper** | Export circuit graph as LaTeX figure |
| **Community** | Share circuits with threaded discussions |
| **Leaderboards** | "Most Interesting Circuit" challenge |
| **GPU Marketplace** | Use rented GPU for circuit extraction on large models |

---

## 11. ADVANCED: WEBGPU COMPUTE SHADER FOR REAL-TIME ATTENTION

For the most performance-critical path—computing attention matrices in real-time—a WGSL compute shader:

```
Workgroup: 8×8 threads
Dispatch: ceil(N/8) × ceil(N/8)
```

Each thread computes one cell of the QK^T matrix, then a second dispatch applies softmax row-wise. This enables real-time attention pattern updates as the user changes input text, without requiring a server round-trip. The WebGPU pipeline handles QKV projections and attention score computation, while CPU threads manage non-matmul operations.

---

## SUMMARY: IMPLEMENTATION ROADMAP

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1: Backend Circuit Extraction** | Weeks 1-2 | TransformerLens + NNsight integration, QK/OV matrix extraction API, head type classification |
| **Phase 2: 2D Circuit Graph** | Weeks 3-4 | Cytoscape.js bipartite graph, edge bundling, supernode support, annotation system |
| **Phase 3: 3D Circuit Graph** | Weeks 5-6 | R3F force-graph, particle flow edges, camera controls, node selection |
| **Phase 4: Intervention Playground** | Weeks 7-8 | pyvene integration, causal tracing, ablation toggles, effect visualization |
| **Phase 5: Browser-Side ML** | Weeks 9-10 | Pyodide + Web Worker inference, WebGPU compute shaders for matrix ops |
| **Phase 6: Platform Integration** | Weeks 11-12 | Connect to experiments, notes, paper, community; sharing and collaboration |