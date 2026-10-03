# 📈 Zone 5: Analytics & Diagnostics Dashboard — Complete Implementation Specification

---

## Overview

Zone 5 is the **quantitative intelligence center** of the attention dashboard. While Zone 2 shows *what* attention looks like and Zone 3 explains *how* it works mathematically, Zone 5 answers *why* certain heads matter, *which* heads are similar, *when* patterns emerge during training, and *where* token meaning travels across layers. It transforms raw attention tensors into **actionable diagnostic metrics** that directly inform model compression, architecture selection, and training monitoring.

The five features in this zone are arranged as a **tabbed panel** occupying the right half of the dashboard (or the full width below the canvas on smaller screens). Each tab is a self‑contained analytical view, but all share a common data pipeline and state management architecture.

---

## Shared Infrastructure for All Zone 5 Features

### A. Data Pipeline Architecture

All five features consume the same underlying data but transform it differently. The pipeline follows a **three‑stage ETL pattern** optimized for browser execution:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  RAW DATA    │────▶│  COMPUTE LAYER  │────▶│  RENDER LAYER    │
│  (Zone 2)    │     │  (Web Worker)   │     │  (Main Thread)   │
│              │     │                 │     │                  │
│ • Attention  │     │ • Head scoring  │     │ • 3D Scatter     │
│   matrices   │     │ • Clustering    │     │ • Dendrogram     │
│ • Hidden     │     │ • Logit lens    │     │ • Timeline       │
│   states     │     │ • UMAP/t‑SNE    │     │ • Trajectories   │
│ • Gradients  │     │ • Time‑lapse    │     │ • Word cloud     │
└──────────────┘     └─────────────────┘     └──────────────────┘
        │                     │                       │
        ▼                     ▼                       ▼
   IndexedDB             Pyodide +              React + Three.js
   (cache)              NumPy/SciPy            + Deck.gl + D3.js
                         in Web Worker
```

**Stage 1 — Raw Data Extraction:**  
Attention matrices and hidden states are extracted from the model via `transformers.js` (running ONNX Runtime in the browser) or fetched from the Django backend API. For models too large for client‑side inference, the backend computes attention and streams results via the existing WebSocket infrastructure or REST endpoints. Raw data is cached in **IndexedDB** using `idb-keyval` for instant recall on repeated explorations.

**Stage 2 — Compute Layer (Web Worker with Pyodide):**  
Heavy numerical computation—singular value decomposition, hierarchical clustering, t‑SNE/UMAP projection, head importance scoring—runs in a **dedicated Web Worker** using Pyodide (CPython compiled to WebAssembly). Pyodide provides NumPy, SciPy, and scikit‑learn directly in the browser, avoiding the need to re‑implement these algorithms in JavaScript. The worker receives raw tensors as JSON/ArrayBuffer, executes Python scripts, and returns computed results via `postMessage`. This keeps the main thread free for rendering and interaction, maintaining 60fps even during heavy computation.

**Stage 3 — Render Layer (Main Thread):**  
Computed results arrive as typed arrays or JSON and are passed to the visualization components. Each feature uses a specialized rendering library chosen for its performance characteristics with the expected data size:

| Feature | Data Size | Rendering Library | Rationale |
|---------|-----------|-------------------|-----------|
| 3D Scatter | 144–1200 points (heads) | Deck.gl (`ScatterplotLayer`) | GPU‑accelerated via WebGL2; handles millions of points; built‑in picking, brushing, viewport |
| Dendrogram | 144–1200 leaves | D3.js (`d3-hierarchy`) + React‑wrapper | SVG rendering for crisp vector output; interactive zoom/pan via `d3-zoom` |
| Logit Lens Timeline | 10–50 layers × top‑k words | Custom Canvas + Framer Motion | Canvas for word cloud rendering; Framer Motion for smooth timeline scrubbing |
| Evolution Time‑Lapse | 144 heads × N epochs | Deck.gl + `requestAnimationFrame` | GPU‑accelerated point updates; smooth animation loop |
| Semantic Pathway | 10–50 tokens × 10–50 layers | Deck.gl + Three.js lines | GPU scatter for points; Three.js for curved trajectory tubes |

### B. State Management

All Zone 5 features share a single **Zustand store** (`useAnalyticsStore`) that holds computed metrics, derived from the raw attention data. This store is separate from the main `useAttentionContext` store (Zone 1) to avoid unnecessary re‑renders of the visualization canvas when only analytics data changes.

```typescript
interface AnalyticsState {
  // Head metrics (Feature 16)
  headScores: HeadScore[];           // importance, stability, similarity per head
  headClusters: ClusterResult | null; // dendrogram data (Feature 17)

  // Logit lens (Feature 18)
  layerPredictions: LayerPrediction[]; // top‑k tokens per layer

  // Evolution (Feature 19)
  evolutionData: EvolutionSnapshot[];  // head scores per epoch
  currentEpoch: number;

  // Semantic pathway (Feature 20)
  tokenEmbeddings: TokenEmbedding[];   // 2D coordinates per token per layer
  tokenTrajectories: Trajectory[];

  // Computation state
  isComputing: boolean;
  computeProgress: number;            // 0–100
  computeError: string | null;
}
```

Components subscribe to **granular selectors** (e.g., `useAnalyticsStore(s => s.headScores)`) to prevent re‑renders when unrelated state changes.

### C. Computation Trigger & Loading States

All five features are computed **lazily**—only when the user opens the corresponding tab for the first time, or when the underlying model/text/layer changes. A **progress indicator** (skeleton shimmer with percentage) appears during computation. If the computation takes longer than 2 seconds, a **"Running in background"** toast appears, allowing the user to continue exploring other zones while analytics compute.

---

## Feature 16: Head Importance × Stability × Similarity 3D Scatter

### 16.1 Scientific Foundation

This feature directly implements the three pruning criteria proposed by BHPVAS (Liu et al., *Journal of Visualization*, 2024): importance score, stability score, and similarity score for evaluating self‑attention heads. These three metrics form a **comprehensive diagnostic framework** where each head can be assessed simultaneously for its contribution (importance), consistency (stability), and redundancy (similarity).

### 16.2 Metric Definitions

**Importance Score:**  
For head `h` in layer `l`, the importance measures how much the model output changes when this head is ablated (zeroed out). It is computed as:

```
I(l,h) = ‖L(x) − L_{¬(l,h)}(x)‖₂
```

where `L(x)` is the original loss (or output logits) and `L_{¬(l,h)}(x)` is the loss after masking head `(l,h)`. In practice, this is approximated using the first‑order Taylor expansion:

```
I(l,h) ≈ ‖∂L/∂A_{l,h} ⊙ A_{l,h}‖₂
```

where `A_{l,h}` is the attention weight matrix for head `(l,h)`. This avoids the computational cost of full ablation. The gradient `∂L/∂A_{l,h}` is obtained during a single backward pass on the calibration text.

**Stability Score:**  
Stability quantifies how consistent a head's attention pattern is across different inputs. Given a set of `K` calibration texts, the attention matrix `A_{l,h}^{(k)}` is computed for each. The stability is:

```
S(l,h) = 1 − σ( { ‖A_{l,h}^{(i)} − A_{l,h}^{(j)}‖_F } ) / μ
```

where `σ` is the standard deviation of pairwise Frobenius distances between attention matrices, and `μ` is their mean. A low stability score indicates a head whose behavior varies wildly with input, while high stability suggests a specialized, consistent function.

**Similarity Score:**  
Similarity measures how redundant a head is with respect to other heads in the same layer. For head `(l,h)`, the similarity is the average Jensen‑Shannon divergence between its attention distribution and every other head in layer `l`:

```
Sim(l,h) = (1/N) Σ_{j≠h} JSD( A_{l,h} ‖ A_{l,j} )
```

where JSD is the Jensen‑Shannon divergence between the row‑wise attention distributions. High similarity indicates a head that could potentially be pruned without significant loss.

### 16.3 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Computation | Pyodide (NumPy/SciPy) in Web Worker | SVD, JSD, pairwise distances—native Python scientific stack compiled to WASM |
| 3D Rendering | **Deck.gl** (`ScatterplotLayer`) | GPU‑accelerated via WebGL2; handles millions of points at 60fps; built‑in picking, brushing, tooltip, viewport controls |
| React Integration | `@deck.gl/react` | Declarative React component for Deck.gl layers |
| Axes & Labels | Deck.gl `TextLayer` + `GridLayer` | GPU‑rendered text for axis labels |
| Orbit Controls | Custom `OrbitController` class wrapping Deck.gl view state | Provides rotate, pan, zoom with inertia |
| Color Mapping | `d3-scale-chromatic` | Continuous color scales (Viridis, Plasma, Turbo) for head type encoding |
| Brushing | Deck.gl `BrushingExtension` | Rectangle selection to filter heads by metric range |

### 16.4 Why Deck.gl Over Plotly or Three.js

Plotly.js 3D scatter plots struggle with more than a few thousand points and lack GPU acceleration for true 3D. React Three Fiber, while powerful, requires manual shader programming for optimal performance with large point clouds. Deck.gl provides the ideal middle ground: a **pre‑built, GPU‑accelerated ScatterplotLayer** designed explicitly for large‑scale data visualization, with built‑in interactivity (hover, click, brush) and seamless React integration via `@deck.gl/react`.

### 16.5 Component Architecture

```
HeadScatter3D (container)
├── DeckGL (canvas)
│   ├── ScatterplotLayer (head points)
│   │   └── Each point: position=[importance, stability, similarity]
│   │                  radius=10, color=by head type, opacity=0.8
│   ├── TextLayer (axis labels)
│   │   └── "Importance →", "Stability →", "Similarity →"
│   ├── GridLayer (reference grid on floor plane)
│   └── BrushingExtension (rectangle selection)
├── ScatterTooltip (floating div, positioned by Deck.gl picking)
│   └── Shows: Layer, Head, Scores, Head Type, "Click to zoom"
├── ScatterLegend (fixed position, bottom‑left)
│   └── Color → Head Type mapping
├── ScatterControls (top‑right overlay)
│   ├── Metric selector: which metric on which axis (swappable)
│   ├── Color by: [Head Type | Layer | Importance | Stability | Similarity]
│   ├── Size by: [Uniform | Importance | Stability]
│   ├── Reset view button
│   └── Export PNG/SVG button
└── SelectedHeadPanel (slides up from bottom when a head is clicked)
    ├── Miniature attention heatmap preview
    ├── Numerical scores
    ├── "Zoom to Head View" button (switches Zone 2 to this head)
    └── "Compare with…" button (opens side‑by‑side comparison)
```

### 16.6 Rendering Pipeline

1. **Data arrives** from Web Worker as `Float32Array` of shape `(N_heads, 3)` for positions plus `(N_heads,)` for colors and sizes.
2. **Deck.gl layer** receives data as an array of plain objects: `[{importance, stability, similarity, layer, head, type, color, size}, …]`.
3. **Vertex shader** (GPU) transforms each point from data space to clip space using the current view matrix.
4. **Fragment shader** (GPU) applies Gaussian blur for depth‑of‑field effect (distant points appear softer).
5. **Picking engine** (GPU) renders an off‑screen framebuffer with unique color IDs per point. On mouse move, the color under the cursor is read back to identify the hovered point with zero CPU overhead.
6. **Brushing** uses the BrushingExtension to render a selection rectangle; selected points are highlighted and an `onBrushEnd` callback filters the dendrogram and other linked views.

### 16.7 Interactive Behaviors

| Interaction | Behavior |
|-------------|----------|
| **Orbit** (left‑drag) | Rotate the camera around the data center |
| **Pan** (right‑drag or shift+left‑drag) | Translate the camera |
| **Zoom** (scroll) | Zoom in/out; Deck.gl handles this natively |
| **Hover** | Tooltip appears showing head ID, layer, exact scores, head type |
| **Click** | Selects the head; opens the SelectedHeadPanel; highlights the head in the dendrogram and Model View |
| **Double‑click** | Zooms the camera to frame the selected head; resets on second double‑click |
| **Brush** (shift+drag) | Rectangle selection; selects all heads within the rectangle; updates dendrogram to highlight selected cluster |
| **Axis drag** | Swap which metric is on which axis (drag "Importance" label to the Y‑axis, etc.) |
| **Color legend click** | Filter by head type (e.g., show only "Positional" heads) |

### 16.8 Performance Optimizations

- **Instanced rendering**: Deck.gl uses GPU instancing to render all points in a single draw call, avoiding per‑point CPU overhead.
- **Level‑of‑detail (LOD)**: Points beyond a certain distance threshold are rendered as smaller dots or skipped entirely, controlled by Deck.gl's `pointRadiusMinPixels` and `radiusMaxPixels`.
- **Data update**: When only colors or sizes change (not positions), Deck.gl updates only the affected GPU buffers without re‑uploading geometry.
- **Memoization**: The `ScatterplotLayer` data array is wrapped in `useMemo` with dependencies on `[headScores, colorBy, sizeBy]`.
- **Web Worker offload**: All scoring computation runs in the worker; the main thread only receives the final array of numbers.

### 16.9 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **No data** (model not yet loaded) | Shows placeholder: "Run a forward pass to compute head scores." with a CTA button |
| **Single layer model** | Z‑axis (Similarity) is disabled; scatter becomes 2D; a note explains "Similarity requires ≥2 heads per layer" |
| **Computation timeout (>30s)** | Worker is terminated; error state shows "Computation timed out. Try a smaller model or fewer calibration texts." with retry |
| **All heads identical** | All points cluster at a single location; a warning banner: "All heads exhibit identical behavior—this model may be untrained or collapsed." |
| **GPU not available** | Falls back to SVG rendering via D3.js (slower but functional); a banner indicates "Software rendering mode" |

### 16.10 Accessibility

- Keyboard navigation: Tab to focus the scatter plot; arrow keys rotate the view; +/- zoom; Enter selects the hovered point.
- `aria-label` on the Deck.gl canvas: "3D scatter plot of attention head scores. X‑axis: Importance. Y‑axis: Stability. Z‑axis: Similarity. {N} heads displayed."
- The SelectedHeadPanel announces via `aria-live="polite"` when a head is selected.
- Color‑blind safe mode: toggle in the controls switches from Viridis to a CVD‑friendly palette (Cividis).

---

## Feature 17: Head Clustering Dendrogram

### 17.1 Scientific Foundation

Hierarchical clustering groups attention heads based on the similarity of their attention patterns. Heads that consistently produce similar attention distributions across multiple inputs are grouped into the same cluster. The resulting dendrogram reveals the **functional organization** of the attention mechanism—for example, BERT models typically exhibit distinct clusters for positional heads (attending to adjacent tokens), semantic heads (attending to related content words), delimiter heads (attending to punctuation/separators), and induction heads (copying previous token patterns).

The clustering uses **agglomerative hierarchical clustering** with Ward's linkage criterion, which minimizes the within‑cluster variance. The input is the pairwise Jensen‑Shannon divergence matrix between all heads, computed across a set of calibration texts.

### 17.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Computation | Pyodide (`scipy.cluster.hierarchy`) in Web Worker | SciPy's `linkage` and `dendrogram` functions are the gold standard for hierarchical clustering; running in Pyodide avoids re‑implementation |
| Dendrogram Rendering | **D3.js** (`d3-hierarchy` + custom `d3-cluster`) | D3's cluster layout produces publication‑quality dendrograms with precise control over node positioning |
| React Integration | Custom `DendrogramView` component using `useRef` + `useEffect` | D3 manipulates the DOM directly within a React‑managed container; React owns the container, D3 owns the internals |
| Zoom/Pan | `d3-zoom` | Built‑in zoom behavior with smooth transitions and touch support |
| Color Coding | `d3-scale-chromatic` | Consistent color scheme with the scatter plot |
| Export | `svg-crowbar` or manual `XMLSerializer` | Extracts the SVG for download as PNG/SVG |

### 17.3 Component Architecture

```
DendrogramPanel (container)
├── DendrogramControls (top bar)
│   ├── Linkage method selector: [Ward | Average | Complete | Single]
│   ├── Distance metric selector: [JSD | Cosine | Euclidean]
│   ├── Color by: [Auto‑label | Layer | Head Type]
│   ├── Expand/Collapse all button
│   ├── Export SVG button
│   └── Search head input (highlights matching leaf)
├── DendrogramSVG (scrollable SVG container)
│   ├── <g class="links"> (the tree branches, as curved paths)
│   ├── <g class="nodes"> (internal nodes, as circles)
│   ├── <g class="leaves"> (leaf nodes = individual heads)
│   │   └── Each leaf: colored rectangle + head label
│   ├── <g class="heatmap-strip"> (optional: mini heatmap next to each leaf)
│   └── <g class="cluster-labels"> (auto‑generated cluster names)
├── ClusterDetailPanel (right sidebar, shown when a cluster is selected)
│   ├── Prototypical attention pattern (the cluster centroid)
│   ├── Head count in cluster
│   ├── Average importance/stability/similarity for the cluster
│   ├── "Prune this cluster" button (exports pruning recipe)
│   └── "Explore cluster in 3D" button (highlights cluster in Feature 16)
└── DendrogramTooltip (follows cursor on hover)
    └── Shows: Head ID, Layer, Head Type, Cluster assignment
```

### 17.4 Data Flow

1. **Raw attention matrices** for all heads across K calibration texts are fetched from the cache or computed.
2. **Web Worker** receives the matrices as a flat `Float32Array`.
3. **Python (Pyodide)** computes the pairwise JSD matrix using `scipy.spatial.distance.jensenshannon`, then runs `scipy.cluster.hierarchy.linkage` with Ward's method.
4. The linkage matrix (a `(N-1)×4` array) and the leaf order are returned to the main thread.
5. **D3.js** receives the linkage data and renders the dendrogram using `d3.cluster()`. The `d3-hierarchy` module implements "several popular techniques for visualizing hierarchical data" including node‑link diagrams.
6. **Auto‑labeling**: The system traverses the dendrogram and assigns cluster labels based on the prototypical attention pattern of each cluster. The labeling uses a rule‑based classifier: if the centroid attention pattern shows a strong diagonal (token attending to itself), it's a "Positional" cluster; if attention is concentrated on semantically related tokens (detected via word embeddings), it's a "Semantic" cluster; if attention is on punctuation/CLS/SEP tokens, it's a "Delimiter" cluster.

### 17.5 Interactive Behaviors

| Interaction | Behavior |
|-------------|----------|
| **Hover leaf** | Tooltip with head details; highlights the corresponding point in the 3D scatter |
| **Click leaf** | Selects the head; switches Zone 2 to that head; opens the SelectedHeadPanel |
| **Click internal node** | Selects the cluster; opens ClusterDetailPanel; highlights all leaves in that cluster |
| **Double‑click internal node** | Collapses/expands the subtree below that node |
| **Brush (drag on heatmap strip)** | Selects a range of heads; filters the scatter plot to show only those heads |
| **Scroll** | Zooms the dendrogram in/out (using `d3-zoom`) |
| **Drag** | Pans the dendrogram |
| **Search** | Highlights the leaf matching the typed head ID; scrolls it into view |
| **Linkage change** | Recomputes clustering (triggers Web Worker) with the new method; dendrogram animates to the new layout |

### 17.6 Performance Optimizations

- **Lazy leaf rendering**: Only leaves visible in the viewport (plus a buffer of 50) are rendered as full elements; off‑screen leaves are placeholder rectangles with the same dimensions to maintain scroll height.
- **Virtualized heatmap strip**: The optional mini‑heatmap next to each leaf uses Canvas rendering for hundreds of leaves, avoiding SVG DOM bloat.
- **Memoized linkage**: If the user switches between linkage methods and then back, the previous result is cached (in the Zustand store) to avoid recomputation.
- **Web Worker reuse**: The Pyodide worker is kept alive between computations; subsequent clustering runs use the already‑initialized Python environment, reducing startup latency from ~3s to ~100ms.

### 17.7 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **Single head in a layer** | That layer's heads are excluded from clustering (or shown as isolated leaves); a note explains |
| **All heads identical** | The dendrogram shows a single deep branch with all leaves clustered at zero distance; warning displayed |
| **Computation failure** | Error state with "Clustering failed. Try a different distance metric or fewer heads." |
| **Very large model (1000+ heads)** | Dendrogram becomes too dense; auto‑switch to "circular dendrogram" layout (using `d3.cluster().size([360, radius])` for radial display) |

### 17.8 Auto‑Labeling Algorithm

The auto‑labeling works as follows:

1. For each cluster, compute the **centroid attention pattern** by averaging the attention matrices of all heads in the cluster (after aligning them via the leaf order).
2. Extract **diagnostic features** from the centroid:
   - `diagonal_strength`: mean of the diagonal entries (token self‑attention)
   - `adjacent_strength`: mean of the first off‑diagonal (attention to immediate neighbors)
   - `delimiter_strength`: attention weight on [CLS], [SEP], punctuation tokens
   - `semantic_strength`: correlation between attention weights and word embedding similarity
3. Classify using a **decision tree**:
   ```
   if diagonal_strength > 0.3: → "Positional"
   elif delimiter_strength > 0.2: → "Delimiter"
   elif semantic_strength > 0.15: → "Semantic"
   elif adjacent_strength > 0.3: → "Local Context"
   else: → "Mixed / Diffuse"
   ```

### 17.9 Accessibility

- The dendrogram SVG includes `aria-label` on each leaf: "Head L{layer}H{head}, cluster {name}, similarity score {score}."
- Keyboard: Arrow keys navigate between leaves; Enter selects; +/- expand/collapse.
- Screen reader announces "Dendrogram updated. {N} clusters identified: {cluster names}."

---

## Feature 18: Logit Lens Prediction Timeline

### 18.1 Scientific Foundation

The Logit Lens (nostalgebraist, 2020) is a training‑free interpretability method that reveals how a transformer model's prediction evolves layer by layer. At each layer, the hidden state `h_l` is projected into vocabulary space using the model's unembedding matrix `W_U`:

```
p_l = softmax(W_U · h_l)
```

This "directly reads out the model's implicit prediction at each layer" without any extra machinery. The logit lens shows that early layers produce confused, diffuse predictions, mid layers gradually converge toward the correct token, and final layers refine the probability distribution.

### 18.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Hidden state extraction | `transformers.js` (ONNX Runtime) or backend API | Extracts hidden states from each transformer layer in the browser or via API |
| Vocabulary projection | Custom WebAssembly module or Pyodide (NumPy) | Matrix multiplication `W_U @ h_l` for all layers; NumPy is sufficient |
| Top‑k extraction | Pyodide (NumPy `argsort`) | Finds top‑k token indices and probabilities |
| Timeline Scrubber | Framer Motion + custom slider | Smooth animation between layers; spring physics for scrubber |
| Word Cloud | Custom Canvas component | Renders variable‑sized text with collision detection |
| Token Highlighting | React state + CSS transitions | Highlights the predicted token in the original input text |

### 18.3 Component Architecture

```
LogitLensPanel (container)
├── TimelineScrubber (top)
│   ├── Layer indicator: "Layer 5 / 12"
│   ├── Slider input (range 0 to N_layers-1)
│   ├── Play/Pause button (auto‑advances through layers)
│   ├── Speed control (0.5×, 1×, 2×)
│   └── Current prediction display: "Model predicts: 'Paris' (87.3%)"
├── PredictionWordCloud (center, main visualization)
│   └── Canvas element rendering top‑k tokens as sized text
├── TokenContributionBar (bottom)
│   ├── Horizontal stacked bar showing which tokens contributed to the top prediction
│   └── Each segment colored by attention head, width proportional to contribution
├── LayerTimeline (vertical bar on the left)
│   ├── 12–48 dots representing layers
│   ├── Current layer dot is highlighted and pulsing
│   ├── Click any dot to jump to that layer
│   └── Color gradient: early layers (cold/blue) → late layers (hot/red)
└── InputTextHighlight (overlaid on the original text display)
    └── Each token colored by its attention to the predicted token
```

### 18.4 Data Flow

1. The model runs a forward pass on the input text. Hidden states `h_0, h_1, …, h_L` are captured at each layer's output (after the attention sub‑layer and FFN, before layer norm).
2. The unembedding matrix `W_U` (also called the LM head) is extracted from the model. For BERT, this is the `cls.predictions.decoder.weight`; for GPT, it's `lm_head.weight`.
3. For each layer `l`, the logit lens projection is computed: `logits_l = W_U @ h_l` (matrix multiplication of shape `[vocab_size, d_model] × [d_model]`).
4. Softmax is applied to get probabilities: `p_l = softmax(logits_l)`.
5. The top‑k (k=20) tokens and their probabilities are extracted.
6. The Canvas word cloud renders each token as text sized proportionally to its probability, positioned using a force‑directed layout to avoid overlaps.

### 18.5 Interactive Behaviors

| Interaction | Behavior |
|-------------|----------|
| **Scrub timeline** | Real‑time update of the word cloud and token contribution bar |
| **Click a word in the cloud** | Highlights that token's probability across all layers (shows a sparkline) |
| **Click a layer dot** | Jumps the timeline to that layer |
| **Play** | Auto‑advances through layers at the selected speed; word cloud animates smoothly |
| **Hover a token in the input text** | Shows which layers attend to that token most strongly |
| **Double‑click a layer** | Switches Zone 2 (Attention Canvas) to show the attention pattern at that layer |

### 18.6 Canvas Word Cloud Rendering

The word cloud uses a custom Canvas implementation for performance:

1. **Layout**: A simple force‑directed algorithm places tokens. Each token is a circle (radius ∝ log(probability)). Tokens repel each other; the canvas center attracts all tokens. The simulation runs for 50 iterations (sufficient for 20 tokens).
2. **Rendering**: `ctx.font` is set proportionally to probability (min 12px, max 48px). `ctx.fillStyle` uses a gradient from the current layer's color. `ctx.fillText()` renders each token at its computed position.
3. **Animation**: When the layer changes, tokens smoothly transition to their new positions and sizes using `requestAnimationFrame` with linear interpolation over 300ms.

### 18.7 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **Model without LM head** (e.g., encoder‑only without pretraining head) | Shows "This model does not have a language modeling head. Logit lens requires an unembedding matrix." |
| **Vocabulary too large (100K+ tokens)** | Top‑k is increased to 50; word cloud switches to "dense pack" layout |
| **All layers predict the same token** | Timeline shows a flat line; info banner: "This model converges early—all layers agree on the prediction." |
| **Early layers produce uniform distribution** | Word cloud shows many tiny, equally‑sized words; a note explains "Early layers have not yet focused on a specific token" |

---

## Feature 19: Attention Pattern Evolution Time‑Lapse

### 19.1 Scientific Foundation

During training, attention heads do not immediately specialize. Research shows that positional patterns emerge first (often within the first few hundred steps), followed by semantic patterns, and finally delimiter/induction patterns. This feature visualizes that **developmental trajectory**, allowing researchers to identify the "critical period" when each head crystallizes into its final function.

The time‑lapse is generated from **training checkpoints** saved during the experiment. For each checkpoint, the attention patterns for all heads are extracted, and the head scores (importance, stability, similarity) are computed. The animation shows how heads move through the 3D metric space over training time.

### 19.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Checkpoint loading | Backend API (`/api/experiments/{id}/checkpoints/`) | Checkpoints are stored on the server; loaded on‑demand |
| Head score computation | Pyodide (NumPy) in Web Worker | Same scoring pipeline as Feature 16, applied per checkpoint |
| Animation | `requestAnimationFrame` + Deck.gl data updates | Smooth 60fps animation; Deck.gl efficiently updates point positions without full re‑render |
| Timeline | Custom React slider + Framer Motion | Scrubbable timeline with keyframe markers |
| Sparklines | D3.js (`d3-shape`) rendered to SVG | Small inline charts showing individual head trajectories |

### 19.3 Component Architecture

```
EvolutionPanel (container)
├── EvolutionTimeline (top)
│   ├── Slider: epoch 0 → epoch N
│   ├── Keyframe markers (vertical lines at significant events: "Head 7 crystallizes", "Loss drops")
│   ├── Play/Pause/Step controls
│   └── Speed selector
├── EvolutionScatter3D (center, reuses Feature 16's Deck.gl component)
│   └── Points animate along trajectories as the timeline scrubs
├── HeadTrajectoryPanel (right sidebar)
│   ├── Selected head's trajectory sparkline (importance over time)
│   ├── "Crystallization epoch": when the head's pattern stabilized
│   ├── "Pruning recommendation": based on final scores
│   └── "Compare with another head" selector
├── CrystallizationTimeline (bottom)
│   ├── Horizontal Gantt‑style chart
│   ├── Each row = one head; each bar = from first emergence to crystallization
│   └── Color‑coded by final head type
└── EventLog (bottom‑right)
    └── Auto‑generated timeline of significant events during training
```

### 19.4 Data Flow

1. **Checkpoint list** is fetched from the backend: `[{epoch: 0, checkpoint_path: …}, {epoch: 10, …}, …]`.
2. For each checkpoint, the Web Worker loads the attention data (from cache or backend), computes head scores, and stores the result.
3. The animation interpolates between consecutive checkpoints. For epochs between saved checkpoints, linear interpolation is used (with a note indicating "interpolated").
4. **Crystallization detection**: A head is considered "crystallized" when its attention pattern's JSD to the final pattern drops below a threshold (default 0.1) and remains below for at least 3 consecutive checkpoints.

### 19.5 Animation Pipeline

The animation loop uses `requestAnimationFrame`:

1. On each frame, the current epoch (float) is computed from the timeline position.
2. For each head, its position `(importance, stability, similarity)` is interpolated between the two nearest checkpoints.
3. The Deck.gl `ScatterplotLayer` data array is updated in place (mutating the positions) and `layer.setChangeFlags({dataChanged: true})` is called to trigger a GPU buffer update.
4. The frame rate is capped at 60fps; if computation takes longer than 16ms, frames are skipped to maintain smooth animation.

### 19.6 Interactive Behaviors

| Interaction | Behavior |
|-------------|----------|
| **Scrub timeline** | Real‑time update of all head positions in the 3D scatter |
| **Click a head** | Pauses animation; shows that head's trajectory sparkline; highlights its row in the crystallization Gantt chart |
| **Click a crystallization event** | Jumps the timeline to that epoch |
| **Compare mode** | Select two heads; their trajectories are shown as colored lines in the 3D space |
| **Export animation** | Renders the full animation as an MP4 (using `MediaRecorder` API on the Deck.gl canvas) |

### 19.7 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **No checkpoints available** | Shows "No training history available. Run an experiment with checkpointing enabled." |
| **Single checkpoint** | Static view; animation controls disabled; note: "Only one checkpoint available. Animation requires at least two." |
| **Head never crystallizes** | Marked as "Unstable" in the Gantt chart; sparkline shows continuing fluctuation |
| **Very long training (10K+ epochs)** | Checkpoints are subsampled (max 200 displayed); slider uses logarithmic scale |

---

## Feature 20: Semantic Pathway Cross‑Layer Trajectories

### 20.1 Scientific Foundation

Semantic Pathway (IEEE VIS 2025) is a visual analytics tool that "reveals how token representations evolve across layers in autoregressive Transformer models" by integrating "layerwise semantic trajectories, attention overlays, and output probability views into a unified interface". The key insight is that each token's hidden state `h_l(token)` can be projected into a 2D space (via UMAP or t‑SNE), and connecting the same token across layers reveals how its **meaning evolves** as information flows through the model.

### 20.2 Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Dimension reduction | **umap-js** (JavaScript UMAP) or Pyodide (`sklearn.manifold.TSNE`) | UMAP runs in the browser via WebAssembly; preserves global structure better than t‑SNE and is faster |
| 2D Scatter | Deck.gl `ScatterplotLayer` | GPU‑accelerated; handles hundreds of points smoothly |
| Trajectory Lines | Deck.gl `LineLayer` or Three.js `Line` via `@react-three/fiber` | GPU‑rendered curved lines between token positions across layers |
| Attention Influence Arcs | Deck.gl `ArcLayer` | GPU‑rendered arcs showing which tokens influenced a token's position change |
| Layer Selector | Custom React slider | Filters which layers are shown |
| Token Labeling | Deck.gl `TextLayer` | GPU‑rendered text labels for tokens |

### 20.3 Component Architecture

```
SemanticPathwayPanel (container)
├── PathwayControls (top bar)
│   ├── Dimensionality reduction: [UMAP | t‑SNE | PCA]
│   ├── UMAP parameters: [n_neighbors, min_dist]
│   ├── Color by: [Token | Layer | POS tag | Attention influence]
│   ├── Show trajectories: [All tokens | Selected token only | None]
│   ├── Show influence arcs: [On | Off]
│   └── Export button
├── PathwayScatter2D (center, main visualization)
│   ├── Deck.gl canvas
│   │   ├── ScatterplotLayer: token positions at each layer
│   │   ├── LineLayer: trajectory paths connecting the same token across layers
│   │   ├── ArcLayer: influence arcs between tokens
│   │   └── TextLayer: token labels
├── TokenSelector (left sidebar)
│   ├── List of all tokens in the input text
│   ├── Click to highlight that token's trajectory
│   └── Color picker for custom trajectory colors
├── LayerSlider (bottom)
│   ├── Range slider: select which layers to display
│   └── Preset buttons: "All layers", "Early only", "Late only"
├── TokenInfluencePanel (right sidebar, shown when a token is selected)
│   ├── List of tokens ranked by influence on the selected token
│   ├── Influence score (JSD between trajectory with and without that token)
│   └── Click to highlight the influencing token's trajectory
└── PathwayLegend (bottom‑right)
    └── Color → Layer mapping (gradient from blue/cold → red/hot)
```

### 20.4 Data Flow

1. Hidden states for all tokens across all layers are extracted (same pipeline as Logit Lens).
2. The hidden states for all tokens at all layers are concatenated into a single matrix of shape `(N_tokens × N_layers, d_model)`. This ensures that UMAP/t‑SNE learns a shared embedding space where the same token at different layers can be compared.
3. **UMAP** (via `umap-js` or Pyodide) reduces this matrix to 2D, producing coordinates `(x, y)` for every token at every layer.
4. The coordinates are split by layer and token, forming trajectories.
5. **Attention influence** is computed: for each token `t` at layer `l`, the change in its UMAP position from layer `l-1` to `l` is attributed to other tokens proportionally to the attention weight from `t` to them at layer `l`.

### 20.5 Trajectory Rendering

Trajectories are rendered as **curved paths** (quadratic Bézier curves) connecting the same token's position at consecutive layers. The curve's control point is offset perpendicular to the line connecting the two points, with the offset magnitude proportional to the distance traveled (creating a visually pleasing "flow" effect).

For the selected token, the trajectory is rendered as a **thick, glowing line** (using Deck.gl's `LineLayer` with `opacity: 1` and a larger `widthScale`). Other tokens' trajectories are rendered as thin, semi‑transparent lines.

### 20.6 Influence Arcs

Influence arcs show how attention from one token "pulls" another token's representation. An arc from token A at layer `l` to token B at layer `l` is rendered if the attention weight `A_{l}(A→B)` exceeds a threshold (default 0.1). The arc's **thickness** is proportional to the attention weight, and its **color** matches the attention head that produced it.

Deck.gl's `ArcLayer` is used for this purpose, as it "offers highly‑flexible API to customize how the data should be rendered" with GPU acceleration.

### 20.7 Interactive Behaviors

| Interaction | Behavior |
|-------------|----------|
| **Hover a point** | Tooltip: token text, layer number, 2D coordinates |
| **Click a point** | Selects that token; highlights its full trajectory; opens TokenInfluencePanel |
| **Click a trajectory** | Selects the token; same as clicking a point on that trajectory |
| **Drag the layer slider** | Filters which layers' points and trajectories are visible |
| **Hover an influence arc** | Tooltip: "Token X attends to Token Y with weight 0.34 (Head 7)" |
| **Double‑click empty space** | Resets selection; shows all trajectories |
| **Scroll** | Zooms the scatter plot |
| **Drag** | Pans the scatter plot |

### 20.8 Performance Optimizations

- **UMAP caching**: The UMAP projection is cached per (model, text) pair in IndexedDB. Re‑loading the same text uses the cached coordinates.
- **Progressive rendering**: Trajectories for the first 5 tokens render immediately; remaining tokens render over subsequent frames to avoid frame drops.
- **Arc culling**: Influence arcs below the threshold are not sent to the GPU, reducing the number of arcs from O(N²) to O(N) in practice.
- **Layer filtering**: When the layer slider filters to a subset of layers, only those points and trajectories are sent to Deck.gl.

### 20.9 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **Single‑token input** | Trajectories view is replaced with a note: "Trajectories require at least 2 tokens to show cross‑token relationships." |
| **UMAP produces overlapping points** | A jitter toggle adds small random offsets to separate overlapping points |
| **Very long text (500+ tokens)** | Only the first 128 tokens are projected; a note explains the truncation |
| **UMAP computation timeout** | Falls back to PCA (faster but less expressive); a banner indicates "Using PCA for faster rendering. Switch to UMAP for better separation." |

---

## Cross‑Feature Integration

### Brushing and Linking

All five features are **linked** through the shared `useAnalyticsStore`. Interactions in one feature propagate to others:

- **Selecting a head in the 3D scatter** highlights that leaf in the dendrogram, shows its trajectory in the evolution time‑lapse, and marks its tokens in the semantic pathway.
- **Selecting a cluster in the dendrogram** filters the 3D scatter to show only those heads, updates the evolution view to show the cluster's collective trajectory, and highlights all tokens influenced by those heads.
- **Scrubbing the logit lens timeline** updates the semantic pathway to show token positions at that specific layer.

### Unified Export

A single "Export Dashboard Report" button in Zone 5 generates a comprehensive PDF containing all five visualizations (as SVG/PNG), the computed metrics table, and an auto‑generated textual summary of findings (e.g., "Layer 5 contains 3 positional heads and 1 semantic head. Head L5H7 shows high importance (0.87) and low similarity (0.12), making it a strong candidate for retention during pruning.").

---

## Summary

Zone 5 transforms raw attention tensors into a comprehensive diagnostic toolkit. The five features work together to answer the full lifecycle of attention analysis: **which heads matter** (3D scatter), **which heads are similar** (dendrogram), **what the model predicts** (logit lens), **how patterns evolve** (time‑lapse), and **where meaning flows** (semantic pathway). Each feature uses specialized rendering technology (Deck.gl for GPU‑accelerated scatter plots, D3.js for dendrograms, Canvas for word clouds) and offloads heavy computation to a Pyodide Web Worker to maintain a responsive 60fps interface.