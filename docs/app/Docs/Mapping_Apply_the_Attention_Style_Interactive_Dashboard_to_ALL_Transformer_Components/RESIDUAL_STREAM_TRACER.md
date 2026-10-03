# 🧠 Residual Stream Tracer — Complete Implementation Specification

## "The Information Highway" — Full Architecture & Implementation Guide

---

## 0. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend Computation** | Python 3.11+ with `nnsight` (`0.4+`) | Primary library for model instrumentation, tracing, and activation capture. Wraps HuggingFace or custom PyTorch models. Provides remote execution via Modal for GPU-accelerated inference with snapshot-based cold-start reduction — exactly as demonstrated by the `mechinterp` webapp which uses FastAPI + Modal for GPU processing |
| **Backend Computation (Alternative)** | `TransformerLens` (`2.0+`) | Alternative to nnsight for HookedTransformer models. Provides built-in `logit_lens()`, `activation_patching()`, `residual_stack_plot()`, and cache-based activation collection |
| **Logit Lens Computation** | `nnsight` tracing context + LM head projection | The nnsight library provides first-class support for Logit Lens: "A classic application: decode each layer's hidden states through the final layer norm and language model head to see what the model 'thinks' at each layer" |
| **Activation Patching** | `nnpatch` library or `pyvene` | `nnpatch` sits on top of nnsight and eliminates manual loop writing for activation patching workflows. `pyvene` provides counterfactual intervention primitives with a "worlds" axis for parallel model runs |
| **Logit Prisms Decomposition** | Custom PyTorch module using frozen activations | The Logit Prisms paper treats non-linear activations as constants, enabling linear decomposition of logit output into per-component contributions via linear projections. Implemented as a post-hoc analysis pass over cached activations |
| **API Layer** | FastAPI (async) + Modal (serverless GPU) | FastAPI serves as the bridge between the React frontend and GPU-accelerated model computation. Modal provides cold-start-optimized GPU inference with snapshot caching |
| **Data Transport** | WebSocket (Django Channels) for streaming + REST for one-shot requests | Real-time residual stream data streams via WebSocket at configurable frame rates; static snapshots via REST. Both transported as compressed JSON with Float32Array optimization |
| **Frontend Rendering — SVG/DOM** | D3.js `v7` + `d3-sankey` plugin + `@visx/sankey` + React `18` | The Sankey diagram layout engine. `d3-sankey` computes the node positions and link paths; `@visx/sankey` provides React component wrappers. D3's `d3-force` module also used for force-directed token relationship graphs |
| **Frontend Rendering — 3D/WebGL** | `react-three-fiber` (`@react-three/fiber` `v9`) + `three.js` `v0.170+` + `@react-three/drei` | Renders the 3D vector arithmetic visualization (tokens as spheres, attention as curved tubes, residual updates as arrow vectors). R3F drives a render loop outside React without overhead, using GPU rendering via WebGL |
| **Frontend Rendering — Particle Systems** | `@newkrok/three-particles` (`v0.4+`) with WebGPU compute shaders + `wawa-vfx` | For the flowing particle attention visualization. GPU compute shaders handle per-particle physics (gravity, velocity, force fields) enabling 50K-350K+ particles at interactive frame rates via Three.js WebGPU renderer and TSL (Three Shading Language) |
| **Large-Scale Network Rendering** | `helios-web` (optional integration) | For datasets with hundreds of thousands of token nodes, Helios-Web provides GPU-based rendering, Web Workers for parallel computing, and customizable force-directed layouts that mitigate the "hairball" effect |
| **2D Vector Arithmetic Rendering** | Canvas API via `react-konva` or raw HTML5 Canvas with Web Worker offloading | For the Visual Sandbox-style 2D training dynamics view. Dimensionality reduction (UMAP/t-SNE) computed in a Web Worker; Canvas renders the 2D point cloud with animated arrows showing vector updates |
| **Dimensionality Reduction** | `umap-js` (in-browser) or `sklearn.umap` (backend via Modal) | For projecting high-dimensional hidden states to 2D/3D for visualization. In-browser UMAP via WebAssembly for small datasets; server-side sklearn for large datasets |
| **State Management** | `zustand` + `@tanstack/react-query` | Zustand stores the visualization state (selected layer, active views, animation parameters); React Query manages server state (fetched activations, cached model outputs) |
| **Animation Orchestration** | `framer-motion` (`v11+`) + `GSAP` (`ScrollTrigger`) | Framer-motion for declarative React animations (view transitions, panel animations). GSAP for high-performance timeline-based animations (scrubbing through layers, animated vector arrows) |
| **Color & Typography** | `d3-scale-chromatic` + Tailwind CSS | D3 provides perceptually uniform color scales (Viridis, Turbo, Cividis) for attention weight mapping. Tailwind handles layout, typography, and dark theme |

---

## Zone 1 — Residual Stream Flow Diagram (Sankey)

### Implementation Architecture

**Backend Data Pipeline:**
1. The model receives input text via a FastAPI endpoint. Using nnsight's tracing context, every intermediate residual stream hidden state is captured at each layer: `model.trace(input_ids)` → access `model.transformer.h[layer].output[0]` (the residual stream output after attention + FFN).
2. The Logit Prisms decomposition is applied: treating non-linear activations as constants, the residual stream at layer L is decomposed into the sum of contributions from every previous component (embedding + attention_0 + FFN_0 + attention_1 + FFN_1 + … + attention_L + FFN_L). Each contribution is computed via a linear projection through the unembedding matrix.
3. For each token position, a "flow bundle" is computed: the magnitude of the residual update from each component (attention head, FFN neuron group) is measured as the L2 norm of the update vector projected to vocabulary space. These magnitudes become the Sankey link widths.
4. The data is serialized as: `{ nodes: [{id, label, layer, componentType}], links: [{source, target, value, tokens}] }` where `value` is the normalized flow magnitude.

**Frontend Sankey Rendering:**
1. The `d3-sankey` library computes the layout. Nodes are organized as vertical columns: Column 0 = Embedding, Columns 1-L = Layer 1 through Layer L (each with sub-nodes for Attention output and FFN output), Final Column = Output Logits. The `@visx/sankey` React wrapper renders the SVG with proper link gradients and node rectangles.
2. Each link is colored by its source component type (blue for attention, green for FFN, orange for embedding). Link opacity is proportional to flow magnitude. Hovering a link highlights all connected nodes and displays the exact flow value in a tooltip.
3. The Sankey diagram supports zoom/pan via D3's `d3-zoom` behavior integrated into the SVG. Clicking a node expands it into its sub-components (e.g., clicking "Attention Layer 3" expands into individual attention heads as separate thin links).

**Performance Optimization:**
- For models with many layers (12+), the Sankey layout is computed asynchronously in a Web Worker to avoid blocking the main thread.
- Link rendering uses SVG `<path>` elements with `stroke-dasharray` animation to create a "flowing" effect (like liquid moving through pipes). The animation is driven by `requestAnimationFrame` with CSS `stroke-dashoffset` animation for GPU-accelerated rendering.
- For very large models (100+ layers), links below a configurable threshold (e.g., 0.1% of max flow) are aggregated into an "Other" category to keep the visualization readable.

---

## Zone 2 — Logit Lens Projection Timeline

### Implementation Architecture

**Backend Computation:**
1. Using nnsight's tracing context, at each transformer layer, the hidden state (after the residual update) is captured. The nnsight library provides this natively: "decode each layer's hidden states through the final layer norm and language model head to see what the model 'thinks' at each layer".
2. For autoregressive models, at each layer the hidden state at the last token position is projected through the final LayerNorm and unembedding matrix (`lm_head`) to obtain a probability distribution over the vocabulary. This produces a `[n_layers, vocab_size]` tensor of log-probabilities.
3. The top-k predictions (k=10, configurable) at each layer are extracted with their probabilities. Additionally, the probability of the *actual* next token (if known) is tracked across layers.
4. Data format: `{ layers: [{ layer_index, top_k: [{ token, token_id, logit, probability }], actual_token_probability }] }`.

**Frontend Timeline Rendering:**
1. The timeline is rendered as a horizontal scrubber bar (like a video timeline) representing layers 0 through N. A draggable "playhead" allows the user to select any layer. Below the scrubber, the current layer's top-k predictions are displayed as a horizontal bar chart, with bar width proportional to probability.
2. The "prediction convergence" animation is the key visual: as the user scrubs through layers, the bars animate smoothly between layer states. At early layers (0-2), the predictions are scattered and low-confidence (many small bars). At middle layers (3-8), the correct prediction starts dominating. At late layers (9-11), the probability distribution sharpens to a single clear answer.
3. A secondary "prediction convergence line chart" shows how the probability of the top-1 token evolves across all layers as a single line, with hover annotations at each data point showing the exact token and probability.
4. Color coding: correct prediction = green, plausible alternatives = yellow, clearly wrong = red. The color intensity scales with probability.

**Performance:**
- The full vocabulary distribution is computed server-side and only the top-k results are transmitted to the frontend (reducing payload from ~50K floats to ~100 floats per layer).
- Layer data is streamed progressively via WebSocket as computation completes, so the user sees results appear layer-by-layer rather than waiting for all layers to finish.

---

## Zone 3 — 2D Training Dynamics Sandbox

### Implementation Architecture

This feature uses the "Visual Sandbox" approach from the Stanford 2024 paper: "with the embedding dimension constrained to d=2, allows for comprehensive two-dimensional visualization of each layer's dynamics" through three phases: memorization, circuit formation, and cleanup.

**Backend:**
1. A small transformer model (d_model=2, 2-4 layers, 2-4 attention heads) is either loaded from a pre-trained checkpoint or trained on-the-fly using a tiny synthetic dataset (e.g., XOR pattern, simple sequence copying).
2. During a forward pass, nnsight captures every hidden state vector (2D) at every token position and every layer. The attention outputs and FFN outputs are also captured as 2D vectors.
3. The residual update formula is visualized: `x_{l} = x_{l-1} + attn_out + ffn_out`. All three vectors (the original position, the attention contribution, and the FFN contribution) are returned for each token at each layer.
4. Data format: `{ tokens: [{ text, positions: [{ layer, x, y, attn_dx, attn_dy, ffn_dx, ffn_dy }] }] }`.

**Frontend Canvas Rendering:**
1. The main canvas (HTML5 Canvas via `react-konva` for declarative React integration) renders the 2D embedding space. Each token is a colored circle positioned at its 2D coordinates at the currently selected layer. The token's text label is displayed above/below the circle.
2. **Vector arrows**: For the selected token, three arrows emanate from its position at the previous layer:
   - A **blue arrow** showing the attention output contribution (attn_dx, attn_dy)
   - A **green arrow** showing the FFN output contribution (ffn_dx, ffn_dy)
   - A **white dashed arrow** showing the net residual update (attn + ffn)
   - These arrows animate in using GSAP's `fromTo` with staggered delays to show the "addition" happening step-by-step.
3. **Layer scrubber**: A slider at the bottom changes the layer. As the slider moves, all token positions smoothly interpolate (using GSAP) from their positions at the old layer to their positions at the new layer, creating a fluid "morphing" animation.
4. **Training time-lapse mode**: For models trained on-the-fly, a separate slider controls "training epoch." The Canvas shows the same tokens but at a specific epoch. An animation loop can play through all epochs, showing how the embedding geometry evolves through the memorization → circuit formation → cleanup phases identified in the Visual Sandbox paper.

**Performance:**
- The 2D vectors are tiny (2 floats per vector), so data transfer is minimal. All computation is done server-side with nnsight.
- Canvas rendering of ~100 tokens with arrows at 60fps is trivial for HTML5 Canvas. The bottleneck is the GSAP animation of arrow appearances, which is GPU-accelerated via CSS transforms.

---

## Zone 4 — Math Panel: Residual Update Formula

### Implementation Architecture

The math panel renders the residual stream update formula as an interactive LaTeX display:

**Formula:** `x_l = x_{l-1} + \text{Attn}(\text{LN}_1(x_{l-1})) + \text{FFN}(\text{LN}_2(x_{l-1}))`

**Implementation:**
1. The formula is rendered using KaTeX (faster than MathJax for real-time updates) in a dedicated panel. Each term in the formula is wrapped in a `<span>` with a unique CSS class for highlighting.
2. When the user hovers over a term (e.g., "Attn(LN₁(x))"), the corresponding visual element in the 2D Canvas (the blue attention arrow) and in the Sankey diagram (the attention flow links) simultaneously highlight with a glow effect. This cross-view brushing is implemented via a shared zustand store.
3. **3D Vector Arithmetic Mode**: For users who want to see the vector math in 3D, the React Three Fiber scene renders the residual stream vectors as 3D arrows. A token's embedding at layer L-1 is a sphere, the attention output is a blue arrow extending from it, the FFN output is a green arrow, and the resulting position is a larger sphere. The camera orbits automatically to show the 3D relationship. All three vectors are labeled with 3D text sprites.
4. **Numerical value overlay**: Below the formula, a table shows the actual numerical values: `‖x_{l-1}‖ = 3.42`, `‖attn‖ = 0.87`, `‖ffn‖ = 1.23`, `cos(attn, ffn) = 0.34`. These update in real-time as the user changes layers or tokens.

**Technology**: KaTeX for LaTeX rendering, react-three-fiber for the 3D vector scene, zustand for cross-component state synchronization.

---

## Zone 5 — Analytics: Residual Stream Norm Analysis

### Implementation Architecture

Three analytics panels provide quantitative insight into the residual stream:

**1. Residual Stream Norm Growth Chart:**
- A line chart (using `recharts` or `visx`) plotting the L2 norm of the residual stream at each layer, averaged across all token positions. Multiple lines can be shown for different input texts (overlay comparison mode).
- Backend: nnsight captures the hidden states and computes per-layer norms. A single API call returns the full norm vector.
- Frontend: recharts `LineChart` with `ReferenceLine` at y=0 and tooltips showing exact values at each layer.

**2. Update Magnitude Distribution:**
- A stacked bar chart showing, for each layer, the relative contribution of attention output vs. FFN output to the total residual update. The ratio ‖attn‖/(‖attn‖+‖ffn‖) is plotted as a percentage.
- This reveals which layers are "attention-dominated" vs. "FFN-dominated."
- Backend: norms are computed from the captured attention and FFN output vectors.

**3. Direction Consistency Matrix:**
- A heatmap (layer × layer) showing the cosine similarity between the residual update direction at layer i and layer j. High off-diagonal values indicate layers that "push" in similar directions, suggesting redundancy.
- Backend: cosine similarities computed from the captured vectors.
- Frontend: rendered as an SVG heatmap using D3, with a Viridis color scale. Hovering a cell shows the exact cosine value.

---

## Zone 6 — Interpretability: Logit Prisms Decomposition & Activation Patching

### Logit Prisms Decomposition

**Backend Implementation (using nnsight + custom PyTorch):**
1. The Logit Prisms method is implemented as a post-hoc analysis. After capturing all intermediate activations via nnsight's tracing context, the key insight is applied: "by treating the model's non-linear activations as constants, we can decompose the output in a linear fashion, expressing it as a sum of contributions".
2. For the residual prism: the residual stream at layer L is expressed as `resid_L = embed + Σ_{i=0}^{L} (attn_out_i + mlp_out_i)`. Each term's contribution to the final logit is computed by multiplying by the unembedding matrix: `logit_contribution_i = unembed @ LN_final(component_i)`. This is a single matrix-vector multiplication per component.
3. For the attention prism: the attention output at layer L is further decomposed into per-head contributions: `attn_out_L = Σ_{h=0}^{H-1} head_out_{L,h} · W_O[:, h*d_head:(h+1)*d_head]`. Each head's logit contribution is computed similarly.
4. For the MLP prism: the MLP output is decomposed into per-neuron contributions using the activation values as constants.

**Frontend:**
1. The "Prism Decomposition View" shows a waterfall chart (horizontal stacked bar) for a selected target token. Each bar segment represents one component's contribution (positive = pushing toward the token, negative = pushing away). Segments are colored by component type (embedding = orange, attention heads = blue shades, MLP neurons = green shades).
2. Clicking a segment drills down: clicking an attention bar reveals per-head sub-contributions. Clicking an MLP bar reveals per-neuron sub-contributions.
3. The total bar length equals the final logit value for the target token.

### Activation Patching

**Backend Implementation (using nnpatch + pyvene):**
1. Two forward passes are performed: a "clean" run (original input) and a "corrupted" run (modified input, e.g., with a key token changed). The nnsight library supports this via its tracing context, and `nnpatch` eliminates the need for manual loop writing by providing a declarative API.
2. For each component (attention head, MLP neuron, layer), the activation from the clean run is "patched" into the corrupted run (replacing the corrupted activation). The effect on the output logit difference is measured.
3. The result is an "attribution matrix": `[component_index, token_position] → Δlogit`. Components with large positive Δlogit are causally important for the clean output.

**Frontend:**
1. The "Patching Results View" displays the attribution matrix as a heatmap (components × token positions), with color intensity proportional to causal effect size.
2. A ranked list shows the top-10 most causally important components for the selected prediction.
3. An interactive mode allows the user to click any component to see a "what-if" comparison: the model's output distribution with and without that component's contribution.

---

## Zone 7 — Social: "Residual Stream Race" Gamified Challenge

### Implementation Architecture

This is a multiplayer competitive game where researchers race to predict the model's output by observing only the residual stream at progressively earlier layers.

**Game Mechanics:**
1. The game presents a model's input text and the residual stream logit lens projection at layer N (starting from the final layer). Players submit their prediction of the next token. Correct predictions earn points; points decrease as N decreases (earlier layers are harder).
2. A leaderboard tracks scores in real-time (using WebSocket for live updates). Players can see each other's predictions appear as they submit.
3. The game has rounds: each round, a new input text is processed by the model. Round 1: layer 11 (easy), Round 2: layer 8 (medium), Round 3: layer 5 (hard), Round 4: layer 2 (expert), Round 5: layer 0 (master — only the embedding!).

**Backend:**
1. The model processes the input once, and all layer projections are cached. Player predictions are compared against the actual top-1 token.
2. Scores are computed server-side and broadcast via WebSocket. The leaderboard is stored in PostgreSQL and updated transactionally.

**Frontend:**
1. The game interface shows the Logit Lens projection timeline (Zone 2) for the current layer. A text input allows players to type their prediction. A countdown timer adds urgency.
2. The leaderboard is displayed as an animated ranking list using framer-motion for position-change animations.
3. Player avatars and usernames are pulled from the existing `profiles` app. Badges earned from the game appear on the player's portfolio.

---

## Data Flow Architecture (Complete)

```
                         ┌─────────────────────────────────┐
                         │        Django Backend            │
                         │  (models, auth, profiles, etc.)  │
                         └──────────────┬──────────────────┘
                                        │ REST API
                         ┌──────────────┴──────────────────┐
                         │     FastAPI Server (GPU)         │
                         │  ┌────────────────────────────┐  │
                         │  │  nnsight Tracing Context    │  │
                         │  │  - Logit Lens per layer     │  │
                         │  │  - Residual stream capture  │  │
                         │  │  - Logit Prisms decompose   │  │
                         │  │  - Activation patching      │  │
                         │  │  - SAE feature extraction   │  │
                         │  └────────────────────────────┘  │
                         │  ┌────────────────────────────┐  │
                         │  │  Modal (GPU Cloud)          │  │
                         │  │  - GPU snapshot caching     │  │
                         │  │  - Cold-start optimization  │  │
                         │  │  - Scaling to multiple GPUs │  │
                         │  └────────────────────────────┘  │
                         └──────────────┬──────────────────┘
                                        │ WebSocket + REST
                         ┌──────────────┴──────────────────┐
                         │       React Frontend             │
                         │  ┌────────────────────────────┐  │
                         │  │  Visualization Layer        │  │
                         │  │  - D3 Sankey (SVG)         │  │
                         │  │  - R3F 3D Vector Scene     │  │
                         │  │  - Canvas 2D Sandbox       │  │
                         │  │  - Recharts Analytics      │  │
                         │  │  - GSAP Animations         │  │
                         │  └────────────────────────────┘  │
                         │  ┌────────────────────────────┐  │
                         │  │  State Layer                │  │
                         │  │  - zustand (view state)    │  │
                         │  │  - React Query (data)      │  │
                         │  │  - WebSocket hook (stream)  │  │
                         │  └────────────────────────────┘  │
                         │  ┌────────────────────────────┐  │
                         │  │  Social Layer               │  │
                         │  │  - Leaderboard (live WS)    │  │
                         │  │  - Comments (community app) │  │
                         │  │  - Badges (reputation app)  │  │
                         │  │  - Game state (WS sync)     │  │
                         │  └────────────────────────────┘  │
                         └──────────────────────────────────┘
```

---

## Key Dependencies (package.json excerpt)

```json
{
  "dependencies": {
    "d3": "^7.9.0",
    "d3-sankey": "^0.12.3",
    "@visx/sankey": "^3.12.0",
    "@visx/responsive": "^3.12.0",
    "react-three-fiber": "^9.0.0",
    "@react-three/drei": "^9.0.0",
    "@react-three/fiber": "^9.0.0",
    "three": "^0.170.0",
    "@newkrok/three-particles": "^0.4.0",
    "react-konva": "^18.0.0",
    "konva": "^9.0.0",
    "framer-motion": "^11.0.0",
    "gsap": "^3.12.0",
    "recharts": "^2.12.0",
    "katex": "^0.16.0",
    "zustand": "^4.5.0",
    "@tanstack/react-query": "^5.0.0",
    "umap-js": "^1.3.0"
  }
}
```

## Key Dependencies (requirements.txt excerpt)

```
nnsight>=0.4.0
transformer-lens>=2.0.0
nnpatch>=0.2.0
pyvene>=0.1.0
fastapi>=0.110.0
modal>=0.60.0
torch>=2.3.0
transformers>=4.40.0
sae-lens>=0.1.0
```

---

## Performance Budgets & Optimization Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| Sankey layout compute | <50ms for 100 nodes | Web Worker with d3-sankey |
| Logit Lens per-layer | <200ms/layer | Modal GPU with snapshot caching |
| 2D Canvas frame rate | 60fps | Canvas with minimal draw calls |
| 3D Vector scene frame rate | 30fps | R3F with instanced meshes |
| Particle system | 50K particles at 30fps | WebGPU compute shaders via @newkrok/three-particles |
| WebSocket latency | <100ms | Django Channels with Redis |
| Data payload per request | <100KB | Top-k selection, Float32Array serialization |
| Cold-start model loading | <5s | Modal GPU snapshots |

This architecture ensures every zone of the Residual Stream Tracer is implemented with production-grade performance, leveraging the most advanced frameworks available in both the Python ML ecosystem and the React/WebGL frontend ecosystem, all deeply integrated with your existing Django platform.