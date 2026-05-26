# 🎨 Zone 2: Attention Canvas — Complete Advanced Implementation Specification

## Overview

Zone 2 is the **central visual real estate** of the entire attention dashboard — occupying approximately 65–75% of the viewport width on desktop (the remaining space divided between Zone 4 Control Panel at left and Zone 5/6 analytics at right). It is the primary space where researchers **see, touch, and manipulate** attention data through six integrated visualization modalities plus two supplementary overlay modes. The canvas operates on a **unified rendering pipeline** that shares attention tensor data, color scales, and interaction state across all views, ensuring that switching between views is instantaneous (under 100ms) and maintains the user's exploration context.

The rendering architecture follows a **layered compositing model** inspired by modern game engines and professional visualization tools like ParaView and Helios-Web. Each visualization view is a self-contained React component that receives a shared `AttentionData` prop from the Zustand store, but internally manages its own WebGL/Canvas/SVG rendering context, camera state, and interaction handlers. A **ViewSynchronizer** middleware ensures that selections (highlighted tokens, focused heads, brushed regions) propagate bidirectionally across all active views.

---

## 1. Rendering Architecture & Performance Foundation

### 1.1 Multi-Backend Rendering Pipeline

Before detailing individual views, Zone 2 is built on a **tiered rendering architecture** that selects the optimal graphics backend based on data complexity, viewport size, and device capability:

| Rendering Tier | Backend | Use Case | Max Data Points |
|---------------|---------|----------|-----------------|
| **Tier 1: SVG** | D3.js + React | Head View (small matrices ≤ 128 tokens), Sankey, Model View | ~16,000 elements |
| **Tier 2: Canvas 2D** | HTML5 Canvas + custom render loop | Head View (medium matrices ≤ 512 tokens), Heatmap, Scatter | ~250,000 elements |
| **Tier 3: WebGL 2.0** | Three.js via React Three Fiber | 3D Topology Map, large scatter plots, particle systems | ~5,000,000 vertices |
| **Tier 4: WebGPU** | Custom WGSL shaders (fallback to WebGL) | Ultra-large attention matrices (>2048 tokens), real-time 3D with compute shaders | ~50,000,000+ vertices |

The pipeline auto-selects the tier based on a **capability detection** routine that runs once on dashboard mount: it probes `navigator.gpu` for WebGPU support, checks `WEBGL_debug_renderer_info` for GPU model, and measures frame timing with a calibration render. The tier can be manually overridden in Zone 4 controls.

Inspired by the GPT2-WebGL project that demonstrated "GPU-accelerated transformer model inference directly in the browser" via WebGL shader programs implementing matrix multiplication, layer normalization, and attention computation, and Helios-Web's approach to "GPU-based rendering for real-time visualization and interactive analysis of massive-scale networks and high-dimensional embeddings".

### 1.2 Shared Color Infrastructure

All views share a unified color system with multiple palettes selectable from Zone 4:

- **Heat (Default)**: A perceptually uniform diverging scale from dark navy (#1e293b) through orange (#ffa94d) to bright red (#ff6b6b), designed using the CIELAB color space for equal perceptual steps.
- **Viridis**: The matplotlib default — perceptually uniform, colorblind-friendly, works in grayscale printing.
- **Blues**: Single-hue sequential for minimalist displays.
- **Spectral**: Full rainbow for detecting fine-grained differences (not recommended for accessibility).
- **Custom**: User-defined via a color ramp editor (drag color stops, save presets).

All color scales are computed as **1D lookup textures** uploaded to the GPU, enabling O(1) color mapping for millions of data points at 60fps.

### 1.3 View Synchronizer — Bidirectional Linked Highlighting

The ViewSynchronizer is the **connective tissue** between all six views. It maintains a shared selection state:

```
SelectionState {
  highlightedTokens: Set<number>       // token indices
  focusedHead: { layer: number, head: number } | null
  brushedRegion: { queryStart, queryEnd, keyStart, keyEnd } | null
  hoveredCell: { query: number, key: number } | null
  activeView: 'head' | 'model' | 'heatmap' | '3d' | 'sankey' | 'qk'
}
```

When a user hovers over token pair (3,7) in the Head View heatmap, the Synchronizer dispatches this event. The 3D View simultaneously highlights the connection between token sphere 3 and token sphere 7 with a glowing pulse. The Sankey View thickens the corresponding flow. The Q/K Joint View enlarges the dots for tokens 3 and 7 and draws a line between them. All views respond within one frame (16ms), creating the illusion of a single, cohesive visualization system.

---

## 2. Feature 4: Head View — BertViz‑Style Interactive Heatmap

### 2.1 Purpose & Core Design

The Head View is the **microscope** of attention analysis — it shows exactly which tokens attend to which other tokens for a specific attention head in a specific layer. It is the most frequently used view (research from BertViz usage patterns shows ~60% of analysis sessions spend most time here). The design follows BertViz's core principle: "an interactive tool for visualizing attention in Transformer language models such as BERT, GPT2, or T5" that "focuses on one or more attention heads within a layer, exposing token-to-token attention weights".

### 2.2 Visual Layout

The heatmap is rendered as an N×N matrix where N = number of tokens. The horizontal axis represents **key tokens** (what is being attended to), and the vertical axis represents **query tokens** (what is doing the attending). Each cell at position (i, j) represents "How much does query token i attend to key token j?"

**Cell rendering**: Each cell is a rectangle (size dynamically computed as `min(48px, max(20px, canvasWidth / N))`). The fill color is computed via the active color scale. Cells also have a subtle **border radius** (2px) for visual softness and a **1px gap** between cells to prevent moiré patterns at high densities. Above 128×128 tokens, cells become 1px with no gap, and the visualization transitions to a **pixel-perfect heatmap** where individual cells are not distinguishable but patterns emerge at the macro level — like viewing a high-resolution thermal image.

**Token labels**: Token text appears along both axes. On the vertical (query) axis, labels are right-aligned. On the horizontal (key) axis, labels are rotated -45° and anchored at the top. Labels are truncated to 8 characters with ellipsis. Hovering a label shows the full token text in a tooltip. Labels use `font-size: 9px`, `font-family: monospace`, `fill: #94a3b8`.

**Diagonal indicator**: The diagonal (where query_index = key_index, representing self-attention) is highlighted with a subtle dashed line at 15% opacity. A toggle in Zone 4 can **mask the diagonal** to reveal off-diagonal patterns more clearly — a common analytical need since self-attention often dominates.

**Color legend**: A horizontal gradient bar at the bottom-right of the heatmap shows the mapping from 0.0 (color-scale minimum) to 1.0 (color-scale maximum). Tick marks at 0, 0.25, 0.5, 0.75, 1.0 with numerical labels.

### 2.3 Interaction Model

The interaction model follows Ben Shneiderman's visual information seeking mantra: **"Overview first, zoom and filter, then details-on-demand."**

**Overview**: The initial render shows the full N×N matrix scaled to fit the viewport. The researcher can immediately see macro patterns — diagonal dominance, block structures, columnar attention (certain key tokens receiving attention from many queries), and row patterns (certain query tokens distributing attention broadly).

**Zoom and Pan**: Scroll wheel zooms in/out centered on the cursor position. The zoom range is 0.25× to 10×. At high zoom levels, individual cells become large enough to display numerical values inside them (values appear at cell size ≥ 32px). Pinch-to-zoom on touch devices. Click-drag pans the viewport. A **minimap** in the corner (like VS Code's minimap) shows the full matrix with the current viewport rectangle indicated, enabling rapid navigation of large matrices.

**Details-on-demand — Hover**: When the cursor hovers over a cell, three things happen simultaneously: (1) the cell gains a white 1.5px stroke and slight scale-up (1.05× for 150ms), (2) a tooltip appears showing: `query_token → key_token : weight = 0.0423`, and (3) the corresponding row and column headers highlight in bold white. The row and column are also semi-highlighted with a 10% white overlay, making it easy to see the full attention pattern for both the query and key tokens.

**Details-on-demand — Click Selection**: Clicking a cell selects the **query token** (the row). The entire row is outlined in white (2px stroke). All other rows dim to 30% opacity. This isolates the attention pattern for one query token, answering "Where is this token looking?" The selected row also triggers a **cross-view highlight** via the ViewSynchronizer — the selected token is highlighted in the 3D View, Sankey, and Q/K Joint View.

**Double-click Selection**: Double-clicking a cell selects both the query AND key tokens, showing the intersection: all query tokens that attend to the selected key token AND all key tokens that the selected query token attends to. This creates a crosshair highlight pattern on the heatmap.

**Range Selection (Brush)**: Click-drag across multiple cells creates a rectangular selection. The selected region is highlighted, and the Synchronizer broadcasts the brushed range to other views. This enables "What's happening in this cluster of attention?" analysis.

**Right-click Context Menu**: Right-clicking any cell opens a context menu with options: "Pin this cell," "Add annotation," "Copy attention value," "Show in Q/K Joint View," "Trace causal path," and "Compare with head X."

### 2.4 Multi-Head Overlay Mode

A toggle in Zone 4 enables **multi-head overlay mode**. Instead of showing a single head, the heatmap shows an **aggregate** of multiple heads. Users select heads via checkboxes in the Control Panel. The overlay uses:

- **Average mode**: Each cell shows the mean attention weight across selected heads. Tooltip shows individual head values: `H3: 0.04, H5: 0.12, H7: 0.08 → avg: 0.08`.
- **Maximum mode**: Each cell shows the maximum attention weight across selected heads, colored by which head contributed the max (a small colored dot in the corner of each cell indicates the winning head).
- **Variance mode**: Each cell shows the variance (or standard deviation) across heads, colored by a separate "agreement" scale — blue for high agreement (low variance), red for disagreement (high variance). This reveals which token pairs different heads agree or disagree on.

### 2.5 Diff Mode

A comparison mode that shows the **difference** between two attention heads (or the same head across two models). Cells are colored by a diverging scale: blue for "Head A attends more," red for "Head B attends more," white for no difference. A summary statistic panel shows: Jensen-Shannon divergence, correlation coefficient, and "top-10 most different token pairs."

### 2.6 Performance Optimization

For large matrices (>256×256), the heatmap uses **level-of-detail (LOD) rendering**:

- **Zoomed out** (viewing the full matrix): Cells are rendered as a single WebGL texture quad with the attention matrix uploaded as a float texture. The GPU interpolates between texels. This renders in a single draw call regardless of matrix size.
- **Zoomed in** (viewing a small region): Only the visible cells are rendered as individual SVG rects or canvas draw calls. The number of rendered elements is bounded by viewport size, not matrix size.
- **Transition**: The transition between LOD levels is seamless — at zoom level 2×, the system cross-fades between the texture-based overview and the geometry-based detail view over 200ms.

This approach, inspired by the GPU-accelerated heatmap generation that "allows for real-time generation of attention maps for 3D models" and GPT2-WebGL's shader-based approach, enables smooth 60fps interaction even with 2048×2048 attention matrices.

### 2.7 Export Capabilities

The Head View supports multiple export formats:

- **SVG**: Vector export for inclusion in LaTeX papers. Includes embedded CSS for consistent rendering. The exported SVG preserves the interactive tooltip data as `<title>` elements.
- **PNG**: High-resolution raster export (up to 4K) for presentations and social media.
- **CSV**: Raw attention matrix data for external analysis in Python/R.
- **JSON**: Full attention data with metadata (model, layer, head, tokens, timestamps) for reproducibility.

---

## 3. Feature 5: Model View — Bird's‑Eye Across All Layers × Heads

### 3.1 Purpose & Core Design

The Model View provides a **macroscopic perspective** — it simultaneously shows every attention head across every layer, enabling researchers to quickly identify patterns, outliers, and structural properties of the attention mechanism as a whole. It follows Dodrio's design principle of "tightly integrating an overview that summarizes the roles of different attention heads".

This view answers questions like: "Which layers have the most specialized heads?", "Are there layers where all heads behave similarly?", "How does head diversity change across layers?", and "Where are the induction heads located?"

### 3.2 Visual Encoding

The Model View is a compact grid where:

- **X-axis**: Head index (0 to H-1, where H = number of heads, typically 12)
- **Y-axis**: Layer index (0 to L-1, where L = number of layers, typically 6 or 12)

Each cell in this grid represents one attention head. The cell encodes **four dimensions** of information simultaneously:

1. **Size (area)** = Head Importance Score. More important heads (those whose ablation causes larger output change) are represented by larger squares. Size ranges from 50% (least important) to 100% (most important) of the cell's available space.

2. **Color (hue)** = Head Specialization Type. A classification algorithm (based on attention pattern clustering) assigns each head to a category: Positional Heads (blue), Semantic Heads (red), Delimiter/Punctuation Heads (green), Induction Heads (gold), Broad-Context Heads (purple), and Unclassified (gray).

3. **Brightness (value)** = Head Diversity Score. How different is this head's pattern from other heads in the same layer? Brighter = more unique/distinctive. Darker = more similar to others (potentially redundant).

4. **Border thickness** = Confidence in the classification. Thicker border = higher confidence (the system is sure this is an induction head). Thin or no border = low confidence (the head's pattern is ambiguous).

### 3.3 Interaction Model

**Hover**: Hovering over a cell shows a tooltip with: Layer X, Head Y, Type: Induction Head (confidence: 87%), Importance: 0.78, Diversity: 0.63, Top attended token pair: "the → cat" (weight: 0.34).

**Hover preview**: A miniature heatmap (64×64px thumbnail) appears as a floating tooltip next to the cursor, showing the actual attention pattern for that head. This allows rapid scanning without clicking — the researcher can move their mouse across the grid and see each head's pattern in quick succession.

**Click**: Clicking a cell zooms into the Head View (Feature 4) for that specific head. The transition is animated: the clicked cell expands from its grid position to fill the canvas, while the grid fades out. This creates a smooth "zoom into detail" experience.

**Shift+Click**: Adds the head to a **multi-select set** (up to 8 heads). Selected heads are highlighted with a glowing border. The user can then choose "Compare Selected" (shows side-by-side heatmaps) or "Overlay Selected" (aggregates them in the Head View).

**Layer-level interactions**: Clicking a layer label on the Y-axis selects all heads in that layer, enabling layer-level analysis. A right-click on a layer label shows layer statistics: mean importance, mean diversity, entropy distribution, and head type composition.

**Brush selection**: Click-drag across multiple cells selects a rectangular region of heads. The analytics panel (Zone 5) updates to show aggregated statistics for the selected region.

### 3.4 Supplementary Views

**Head Similarity Matrix**: A toggle switches the Model View to show an H×H matrix (where H = total heads across all layers = L × H_per_layer). Each cell shows the similarity (1 - Jensen-Shannon divergence) between two heads' attention patterns. This reveals clusters of similar heads and identifies truly unique heads. The similarity matrix is reorderable via hierarchical clustering — drag a dendrogram to reorder rows/columns.

**Head Evolution Over Layers**: A second toggle shows how individual head "positions" change across layers. Heads are tracked by their pattern fingerprint — e.g., "Head 3 in Layer 0 has a 0.87 similarity to Head 7 in Layer 1, suggesting it may have shifted position." This is visualized as connecting lines between corresponding heads across layers, like a parallel coordinates plot.

**Attention Type Comparison Overlay**: When comparing two models (e.g., standard MHA vs. Galerkin), the Model View splits into two grids side-by-side. Differences are highlighted: heads that exist in one model but not the other, heads whose importance changed significantly, and heads whose specialization type changed. The Diff Mode (from Feature 4) can be triggered by clicking any head pair.

### 3.5 Performance Considerations

The Model View grid for a 12-layer, 12-head model contains only 144 cells — rendering is trivial even with SVG. The complexity comes from the hover preview thumbnails, which require generating miniature heatmaps on-the-fly. These are pre-computed when the Model View is opened and cached as 64×64px PNG data URLs (stored in a Map keyed by layer+head). With 144 heads, this consumes approximately 144 × 4KB = 576KB of memory, well within budget.

---

## 4. Feature 6: 3D Attention Topology Map — React Three Fiber

### 4.1 Purpose & Core Design

The 3D Attention Topology Map transforms the flat, abstract attention matrix into an **immersive spatial experience** where tokens become tangible objects and attention connections become visible forces. This view is designed to leverage human spatial reasoning — our brains are wired to understand 3D relationships, making complex attention patterns more intuitive.

The design is inspired by TensorSpace, which "provides Keras-like APIs to build deep learning layers, load pre-trained models, and generate a 3D visualization in the browser" using Three.js, and by the Reddit "Planet Neuron" project where each neuron becomes a planet in a navigable game world. The Helios-Web project further demonstrates that "GPU-based rendering for real-time visualization and interactive analysis of massive-scale networks and high-dimensional embeddings" is feasible in browser environments.

### 4.2 Scene Architecture

The 3D scene is organized as a **multi-layered cylindrical space**:

**Layer Rings**: Each transformer layer is represented as a horizontal ring (torus) in 3D space. Layers are stacked vertically with equal spacing (2.0 world units apart). The first layer (Layer 0) is at the bottom, and the last layer is at the top. Rings are semi-transparent (opacity 0.15) with a subtle glow.

**Token Spheres**: Within each layer ring, tokens are positioned as spheres distributed around the ring's circumference. The angular position of each token on the ring is determined by a dimensionality reduction of its hidden state at that layer (t-SNE or UMAP, computed once when the view loads and cached). This means tokens with similar meanings cluster together on the ring.

Token sphere properties:
- **Size**: Proportional to the token's L2 norm (activation magnitude) at that layer. Important tokens are larger.
- **Color**: By token position in the sequence (rainbow gradient from blue for early tokens to red for late tokens) OR by part-of-speech (noun=blue, verb=red, adjective=green, etc.) OR by semantic cluster (auto-assigned via k-means on hidden states). The coloring mode is selectable from Zone 4.
- **Emissive glow**: Tokens that are currently "attending strongly" to other tokens glow brighter.

**Attention Connections**: Between token spheres, attention weights are visualized as curved **tubes/arcs** (using Three.js `TubeGeometry` along a quadratic Bezier curve). The arc rises above the ring plane, creating a visible "web" of connections:
- **Thickness**: Proportional to attention weight (logarithmic scale, range 0.02 to 0.3 world units).
- **Color**: Matches the source (query) token's color, with opacity proportional to attention weight.
- **Animation**: Active connections have particles flowing along the tube from query to key, at a speed proportional to the attention weight. High-attention connections "flow" faster, creating a dynamic, living visualization.

### 4.3 Multi-Layer 3D Visualization

Unlike the Head View which shows one layer at a time, the 3D View can show **all layers simultaneously**. This is the killer feature: the researcher can physically see how attention patterns **evolve across layers**. A token that receives diffuse attention in Layer 0 might become sharply focused on a specific other token by Layer 6 — this journey is visible as the tube thickness and direction changes across the vertical stack.

**Cross-layer connections**: When a token in Layer 3 strongly attends to a token in Layer 3 (same-layer attention), the tube is horizontal. When showing cross-layer connections (e.g., residual stream flow), tubes angle upward from lower to higher layers.

### 4.4 Camera & Navigation

The camera system supports multiple navigation modes:

**Orbit Mode (Default)**: The camera orbits around the center of the layer stack. Left-drag rotates, scroll zooms, right-drag pans. The orbit target can be set by double-clicking any token sphere — the camera smoothly animates to center on that token.

**Fly-Through Mode**: WASD keys move the camera freely through the 3D space. Q and E roll. This enables "flying through" the attention web — the researcher can navigate inside the model, seeing attention connections from the perspective of individual tokens.

**Layer-Focus Mode**: Pressing a number key (0-9) instantly animates the camera to focus on that layer, with all other layers dimmed to 20% opacity. This enables rapid layer comparison without losing spatial context.

**Preset Views**: Six preset camera angles accessible via buttons: "Top Down," "Side View," "Front View," "Isometric," "Token's Eye" (camera positioned at a selected token, looking outward at what it attends to), and "Global Overview" (camera far back, showing all layers).

### 4.5 Interactive Token Selection

Clicking a token sphere triggers a system-wide event via the ViewSynchronizer:
1. The selected token sphere pulses (scale animation: 1.0 → 1.3 → 1.0 over 600ms) and glows brighter.
2. All attention connections FROM this token are highlighted (other connections dim to 10%).
3. The Head View (if visible) highlights the corresponding row.
4. The Q/K Joint View centers on this token's vectors.
5. A floating info panel appears near the token showing: token text, position, layer, top-3 attended tokens, and attention entropy.

**Double-click** a token to "lock" the selection. **Shift+click** to add to multi-select (up to 5 tokens). Multi-selected tokens show all connections between them — useful for analyzing "How does the subject, verb, and object interact through attention?"

### 4.6 Animation & Time-Based Features

**Attention Flow Animation**: Connections are not static — particles continuously flow from query tokens to key tokens along the attention tubes. This gives the visualization a living, breathing quality. The flow speed for each connection is proportional to its attention weight. Users can pause the flow with Spacebar to inspect a static snapshot.

**Layer Propagation Animation**: A special animation mode (triggered by the "Propagate" button) shows how a single input token's influence spreads through the network. Starting from the selected token at Layer 0, the animation "lights up" tokens at Layer 1 that receive attention from it, then Layer 2 tokens that receive attention from those Layer 1 tokens, and so on — like watching ripples spread across a pond. Each step takes 500ms, with the full propagation through 12 layers taking 6 seconds.

**Timeline Scrubber**: For autoregressive models, a timeline at the bottom shows generation steps (Token 1, Token 2, ..., Token N). Scrubbing the timeline shows the 3D scene at that generation step — new tokens appear, connections re-form, and the spatial arrangement updates.

### 4.7 Visual Effects & Atmosphere

The 3D scene incorporates several atmospheric effects to enhance immersion:

**Particle System Background**: A subtle starfield of 500-1000 slowly drifting particles fills the background, giving depth cues and a sense of vastness. Particle density increases near highly active attention connections.

**Bloom Post-Processing**: An Unreal Bloom pass (via `@react-three/postprocessing`) adds a soft glow to bright attention connections. The bloom threshold is set so only connections with weight > 0.3 glow, creating a natural visual hierarchy.

**Ambient Occlusion**: A subtle SSAO pass adds depth to the token spheres and helps distinguish overlapping connections.

**Fog**: Exponential fog with density 0.002 provides depth perception — distant layers fade slightly into the background.

### 4.8 Performance & LOD

The 3D view implements aggressive LOD:

- **Distance-based**: Token spheres beyond a certain camera distance are rendered as simpler geometries (icosahedron with 2 subdivisions instead of 4). Attention tubes beyond medium distance become simple lines.
- **Frustum culling**: Only objects within the camera's view frustum are rendered.
- **Instanced rendering**: Token spheres across all layers are rendered via `InstancedMesh` — a single draw call for all spheres, with per-instance transforms and colors passed via buffer attributes.
- **Web Workers**: The t-SNE/UMAP computation for token positioning runs in a Web Worker, avoiding main thread blocking. The initial positioning for 128 tokens across 12 layers takes approximately 2-3 seconds and is cached.

---

## 5. Feature 7: Sankey Attention Flow Diagram

### 5.1 Purpose & Core Design

The Sankey Attention Flow Diagram visualizes the Transformer as a **dynamical system** — showing how information flows from input tokens, through attention heads at each layer, and ultimately to output predictions. The Transformer Explainer pioneered this approach: it "adopts the Sankey diagram visual design, inspired by recent work viewing Transformers as dynamical systems, emphasizing how input data flows through model components" and "the Sankey component visualizes data flow between model components using SVG paths and gradients".

### 5.2 Visual Structure

The Sankey diagram is organized as a **left-to-right flow**:

```
Input Tokens ──→ Layer 0 Heads ──→ Layer 1 Heads ──→ ... ──→ Layer L-1 Heads ──→ Output
   (N nodes)     (H nodes)        (H nodes)                 (H nodes)          (N nodes)
```

Each column represents a stage in the processing pipeline. Between adjacent columns, **flow paths** connect nodes with thickness proportional to attention weight (or information flow magnitude).

### 5.3 Node Design

**Input Token Nodes (leftmost column)**: Each input token is a rounded rectangle showing the token text (truncated to 8 characters). Node height is proportional to the token's total "outgoing attention" (sum of its attention to all keys). Nodes are colored by token position (same rainbow as the 3D view).

**Head Nodes (middle columns)**: Each attention head is a rounded rectangle labeled "H3" (for Head 3). Node height is proportional to the head's importance score. Nodes are colored by head specialization type (blue=positional, red=semantic, etc.).

**Output Nodes (rightmost column)**: For encoder models, the output nodes mirror the input tokens. For autoregressive models, the output shows the predicted next token(s), with node height proportional to prediction probability.

### 5.4 Flow Path Design

Flow paths between columns are rendered as **curved SVG paths** using custom path generators. Each path represents the aggregate attention flow from one source node to one target node.

**Path thickness**: Proportional to the attention weight, logarithmically scaled. The minimum visible thickness is 1px (for weights below 0.01), and the maximum is 20px (for weights above 0.5).

**Path color**: Gradient from source node color (at the left) to target node color (at the right), creating a smooth visual transition that helps the eye track flow direction.

**Path opacity**: By default, all paths are rendered at 60% opacity. Hovering a path increases its opacity to 100% and brings it to the front (z-index), while dimming all other paths to 15%.

**Path curvature**: Paths use a custom cubic Bezier curve generator that produces smooth, aesthetically pleasing curves. The curvature parameter is adjustable from Zone 4 (straight lines → gentle curves → dramatic swoops).

### 5.5 Interaction Model

**Hover on node**: Highlights all incoming and outgoing flows for that node. Other flows dim. A tooltip shows: "Layer 3, Head 7 | Receives from: token 'the' (0.12), token 'cat' (0.08) | Sends to: token 'sat' (0.15), token 'mat' (0.11)."

**Hover on flow**: The flow path brightens and thickens slightly. A tooltip shows: "token 'the' → Head 3 → token 'cat' | Weight: 0.23 | Rank: #4 of 144 flows from 'the'."

**Click on node**: Selects that node (highlighted with a glowing border). The ViewSynchronizer propagates this selection — the Head View jumps to that layer/head, the 3D View centers on that token.

**Click on flow**: "Pins" that flow (rendered in gold, always on top). Multiple flows can be pinned for comparison. A pinned flow remains highlighted even when hovering elsewhere.

**Drag to reorder nodes**: Nodes within a column can be vertically reordered by drag-and-drop. The default ordering is by token position (left column) or head index (middle columns), but reordering by "total flow magnitude" or "entropy" can reveal structural patterns.

### 5.6 Animation

**Flow animation**: Animated particles (small circles) travel along the flow paths at constant speed, creating a sense of movement and direction. The animation can be toggled on/off and the speed adjusted.

**Layout transition animation**: When switching between models or attention types, nodes smoothly transition to their new positions (using FLIP animation — First, Last, Invert, Play) over 600ms with easing.

### 5.7 Scalability

For models with many heads (e.g., 32-head architectures) or many layers, the Sankey diagram can become cluttered. Mitigation strategies:

- **Collapse layers**: A range slider lets the user select which layers to show (e.g., "Show layers 0-5 only").
- **Filter by head type**: Show only positional heads, or only semantic heads.
- **Aggregate heads**: Multiple heads in the same layer can be aggregated into a single "meta-node" showing total flow.
- **Minimum flow threshold**: Flows below a certain weight (adjustable slider, default 0.02) are hidden.

---

## 6. Feature 8: Query/Key Joint Embedding Explorer — AttentionViz‑Style

### 6.1 Purpose & Core Design

The Q/K Joint Embedding Explorer visualizes the **geometric relationship** between queries and keys. The key insight from AttentionViz (VIS 2023) is that "the main idea behind our method is to visualize a joint embedding of the query and key vectors used by transformer models to compute attention" — and that "unlike previous attention visualization techniques, our approach enables the analysis of global patterns across multiple input sequences".

Instead of showing the attention *result* (weights), this view shows the attention *computation* — the raw Q and K vectors that are multiplied to produce attention scores. This reveals *why* certain tokens attend to others: because their Q and K vectors point in similar directions (high dot product).

### 6.2 Dimensionality Reduction

Q and K vectors live in a high-dimensional space (typically 64 dimensions for each head). To visualize them in 2D, the system applies dimensionality reduction:

**PCA (Principal Component Analysis)**: Fast (O(d²n)), deterministic, preserves global structure. Best for initial exploration.

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Slower (O(n²)), non-deterministic, preserves local structure. Best for revealing clusters.

**UMAP (Uniform Manifold Approximation and Projection)**: Medium speed, preserves both local and some global structure. Best for detailed analysis.

The reduction method is selectable from Zone 4. The computation runs in a Web Worker and results are cached. For 128 tokens with d=64, all three methods complete in under 2 seconds.

### 6.3 Visual Encoding

The main view is a **2D scatter plot** where each point represents either a query vector or a key vector for a specific token:

- **Query vectors (Q)**: Rendered as circles (●). Color-coded by token position (same rainbow scale).
- **Key vectors (K)**: Rendered as triangles (▲). Color-coded by token position.

Token labels appear next to points (showing the first 6 characters of the token text). Labels can be toggled on/off.

**Background color**: Each point has a "halo" — a subtle radial gradient showing its attention magnitude (sum of attention weights involving this token).

**Connections**: When a query token is selected (clicked), lines are drawn from its Q point to all K points, with line opacity proportional to the attention weight. This reveals the geometric interpretation of attention: a query attends strongly to keys whose vectors are nearby in the embedding space.

### 6.4 Brushing and Linking

The core interaction pattern is **brushing and linking** — a fundamental technique in visual analytics where selection in one view propagates to all linked views.

**Brush**: Click-drag to draw a lasso or rectangular selection on the scatter plot. All Q and K points within the selection are highlighted.

**Link**: The ViewSynchronizer propagates the selection:
- The Head View highlights the corresponding rows (for selected Q) and columns (for selected K).
- The 3D View highlights the corresponding token spheres and attention connections between selected tokens.
- The Sankey View highlights flows involving selected tokens.
- The Analytics panel (Zone 5) computes statistics for the selected subset.

**Zoom**: Scroll to zoom, centered on the scatter plot. Double-click to reset zoom.

**Hover**: Hovering a point shows a tooltip with the full token text, its vector norm, and its "attention footprint" (how broadly it distributes or receives attention).

### 6.5 Multi-Head Mode

A selector in Zone 4 enables showing Q/K vectors for **multiple heads simultaneously**. Each head's Q and K vectors are projected into the same 2D space using a joint dimensionality reduction (computed on the concatenated Q and K matrices across all selected heads). Different heads use different point shapes: Head 0 = ●/▲, Head 1 = ■/▼, Head 2 = ◆/◀, etc.

This reveals how different heads partition the embedding space — e.g., Head 0 might separate tokens by position (early tokens cluster together), while Head 3 separates tokens by semantic role (subjects cluster together).

### 6.6 Trajectory Mode

A unique feature: for a selected token, draw its **trajectory across layers**. The Q vector for token "cat" at Layer 0, Layer 1, ..., Layer L-1 are connected by a path with arrows showing the direction of change. This reveals how a token's "query intent" evolves as it passes through the transformer — does it become more focused? Does it shift from attending to nearby tokens to attending to semantically related tokens?

---

## 7. Feature 9: Token‑by‑Token Generation Animation

### 7.1 Purpose & Core Design

For autoregressive models (GPT-style), this view animates the **step-by-step generation process**, showing how attention patterns evolve as each new token is generated. Semantic Pathway (IEEE VIS 2025) is the primary inspiration: it "integrates layerwise semantic trajectories, attention overlays, and output probability views into a unified interface, enabling users to trace how meaning accumulates and decisions emerge during generation".

### 7.2 Timeline & Playback Controls

The primary control is a **horizontal timeline** at the bottom of the view, showing generation steps as tick marks:

```
[Step 1]──[Step 2]──[Step 3]──...──[Step 20]──[Step 21 (current)]
```

**Playback controls**: Play/Pause, Step Forward (→), Step Backward (←), Speed (0.25×, 0.5×, 1×, 2×, 4×), Jump to Start, Jump to End.

**Scrubber**: Drag the playhead to any position to instantly jump to that generation step. The visualization updates reactively.

**Step indicator**: Shows "Generating token 15 of 50 | Current token: 'cat' | Top-5 predictions: cat (0.34), dog (0.12), bird (0.08), ..."

### 7.3 Visualization Modes During Animation

The generation animation can be viewed through any of the other five visualization modes, creating a powerful multi-perspective analysis:

**Heatmap mode**: The attention matrix grows as new tokens are added. At step 5, the matrix is 5×5. At step 20, it's 20×20. The new row (for the latest token) is highlighted with a brief pulse animation (300ms). The researcher can see how the latest token distributes its attention.

**Sankey mode**: New nodes appear at each column as tokens are generated. Flow paths thicken/thin as attention patterns shift. This is particularly effective for seeing how earlier tokens' attention to later tokens evolves.

**3D mode**: New token spheres materialize at each layer ring with a "spawn" animation (scale from 0 to 1 over 400ms). New attention connections form as curved tubes growing from the new token to existing tokens.

**Q/K Joint mode**: New Q and K points appear in the scatter plot. The trajectory of a specific token's Q vector across generation steps can be traced — does its "query intent" change as more context becomes available?

### 7.4 Semantic Pathway Overlay

A unique overlay inspired by Semantic Pathway: for each token, draw its **cross-layer trajectory** — how its hidden state evolves across layers during a single generation step. The trajectory is a colored path (rainbow from Layer 0 = blue to Layer L = red) in a 2D t-SNE projection. Hovering a trajectory point shows the token's nearest vocabulary words at that layer ("At Layer 4, this token's hidden state is closest to the word embedding of 'feline'").

### 7.5 Output Probability View

Below the main visualization, a **bar chart** shows the model's output probabilities for the next token. As generation progresses, the bar chart updates to show predictions for the current position. The bar for the token that was *actually* generated (in the real model's output) is highlighted in gold, while other top predictions are shown in gray. This enables comparison: "The model predicted 'cat' with 34% probability, but the next most likely token was 'dog' at 12%."

### 7.6 Intervention Mode

A powerful feature for causal analysis: at any generation step, the user can **intervene**:

- **Force a different token**: Instead of the model's top prediction, the user selects a different token from the probability chart. The animation continues with this forced token, showing how the "alternate reality" unfolds.
- **Ablate a head mid-generation**: Toggle off a head and see how the remaining generation differs. The "alternate path" is shown as a ghost overlay (semi-transparent) alongside the original path.
- **Modify attention**: Click a cell in the heatmap and manually set its attention weight to 0 or 1. The generation continues with this modified attention, revealing the causal impact.

---

## 8. Supplementary View Modes — Added Modules

### 8.1 Chord Diagram View (Supplementary Module A)

The Chord Diagram represents attention as a **circular layout** where tokens are arranged around a circle and attention connections are drawn as chords (arcs) connecting them. This is particularly effective for small numbers of tokens (≤20) where individual connections are distinguishable.

**Visual encoding**: Token nodes are evenly spaced around the circle. Each token's "total attention received" is shown as an arc along the circumference (thicker arc = receives more attention). Attention connections are chords colored by the source token's color, with opacity proportional to weight.

**Interaction**: Hovering a chord highlights both connected tokens and shows the attention weight. Clicking a token filters to show only connections involving that token.

### 8.2 Particle Flow View (Supplementary Module B)

An advanced, visually stunning alternative to the static heatmap. Attention is rendered as **flowing particles** — thousands of tiny, glowing dots that stream from query tokens to key tokens along curved paths. This is inspired by particle system effects in modern games and the dynamic visualization approach of the Transformer Explainer's animated Sankey paths.

**Particle properties**: Each particle has a lifetime of 2-3 seconds (configurable). Density of particles along a path is proportional to attention weight. Particle color matches the query token. Particles leave a subtle **glow trail** that fades over time.

**Interaction**: The particle flow can be "frozen" (all particles stop in place) to inspect the static pattern. A "wind" slider adjusts the flow speed. Individual paths can be "seeded" (boost particle density) by clicking a token pair.

---

## 9. Cross-View Synchronization Protocol

The ViewSynchronizer operates on a **pub/sub event bus** implemented via a Zustand store slice. Each view subscribes to selection events and publishes its own. The protocol avoids circular updates through event source tracking:

```
Event { type, sourceView, payload, timestamp }
```

When View A publishes a selection event, Views B through F receive it and update their own highlighting. Views B through F do NOT re-publish the event (detected by matching timestamp and sourceView), preventing infinite loops.

The synchronization latency target is < 16ms (one frame), achieved by using direct store mutations rather than React state updates for cross-view highlighting data.

---

## 10. Accessibility Across All Views

All views implement:

- **Keyboard navigation**: Tab between interactive elements, arrow keys for fine-grained control (move selection one token at a time), +/- for zoom.
- **Screen reader support**: `aria-label` attributes on all interactive elements. The heatmap cells include `aria-label="Attention from token 'The' to token 'cat': 0.0423"`.
- **High-contrast mode**: A toggle in Zone 4 switches all color scales to high-contrast alternatives optimized for visual impairments.
- **Reduced motion**: Respects the `prefers-reduced-motion` media query — animations are disabled or simplified when this is set.
- **Focus indicators**: All interactive elements have visible focus rings (2px solid white outline with 2px offset).

---

## 11. Summary

Zone 2 provides six primary visualization modalities (Head View, Model View, 3D Topology Map, Sankey Flow, Q/K Joint Explorer, Token-by-Token Animation) plus two supplementary modes (Chord Diagram, Particle Flow), all synchronized through a unified rendering pipeline and shared selection state. The tiered rendering architecture (SVG → Canvas → WebGL → WebGPU) ensures smooth performance from small matrices to massive 2048×2048 attention grids. Every view supports multi-head overlay, diff mode, and bidirectional linked highlighting. The zone is fully accessible, responsive, and optimized for both rapid exploration and deep analysis.