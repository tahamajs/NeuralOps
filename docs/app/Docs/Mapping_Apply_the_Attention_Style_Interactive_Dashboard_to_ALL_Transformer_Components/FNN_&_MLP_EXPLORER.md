# 🧠 FFN/MLP EXPLORER — Complete Implementation Blueprint

## "Neuron Activation Universe"

---

## 📐 OVERALL ARCHITECTURE

The FFN/MLP Explorer follows a **six-tier architecture** identical to the approach used by OpenAI's Transformer Debugger: a React frontend, a FastAPI activation server, and a specialized model inference library with hooking capabilities. This architecture separates concerns into: Model & Inference System → Activation Data Management → Activation Server → Frontend Canvas.

The core data abstraction is the `ActivationRecord` (a `@dataclass` pairing activation values with token strings) and the `NeuronRecord` (organizing multiple activation records per neuron with statistical moments: mean, variance, skewness, kurtosis). The `NeuronId` uniquely identifies each neuron by `(neuron_index, layer_index)`. This simple but powerful data model underpins every visualization zone.

---

## 📊 ZONE 1: CANVAS — Neuron Activation Heatmap (`tokens × neurons`)

### Implementation Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Activation Extraction** | `TransformerLens` 3.0+ via `TransformerBridge` | Supports 50+ architectures, exposes all internal activations through hook points including `mlp.post_act` |
| **Backend Server** | FastAPI (Python 3.11+) with `asyncio` | OpenAI's TDB uses FastAPI activation server; AXON uses the same pattern with WebSocket streaming |
| **Data Format** | Apache Arrow via `pyarrow` | Zero-copy columnar transfer between Python and JavaScript; critical for matrices with up to 16384 neurons × 1024 tokens (16M values) |
| **Frontend Rendering** | `@deck.gl/layers` (BitmapLayer + ColumnLayer) | GPU-powered via WebGL2/WebGPU; renders millions of rectangles at 60fps using instanced rendering |
| **Color Scale** | Custom GLSL fragment shader (Viridis/Inferno) | Bypasses CPU color computation; all color mapping happens on GPU |
| **Zoom/Pan** | `deck.gl` built-in `OrthographicView` with `viewState` | Handles coordinate transforms in WebGL without re-rendering data |

### Implementation Details

**Backend Pipeline (FastAPI)**:

```
POST /api/ffn/activations
Body: { model_id, text, layers: [0..N] }
→ TransformerLens runs forward pass with hooks on mlp.post_act for each layer
→ ActivationCache stores all values
→ For each layer, extract tensor [n_tokens, d_mlp] as float32
→ Convert to Arrow RecordBatch (columns: token_index, neuron_index, activation_value)
→ Return as Arrow IPC stream (binary)
```

The model system uses `TransformerHooks` to intercept activations at `mlp.post_act` for each transformer block. The activation server exposes API endpoints for accessing model activations and managing inference operations.

**Frontend Rendering Pipeline**:

```
1. Receive Arrow IPC stream → deserialize into TypedArray (Float32Array)
2. Upload to GPU as SSBO (Shader Storage Buffer Object) via WebGL2
3. Vertex shader: each instance = one (token, neuron) pair; position computed from indices
4. Fragment shader: sample Viridis colormap from 1D texture based on activation value
5. Instanced rendering: single draw call for entire heatmap (up to 16M quads)
6. Hover detection: render separate "picking layer" with encoded (token_idx, neuron_idx)
```

For real-time updates (e.g., changing input text), use `deck.gl`'s `data` prop reactivity — only the SSBO buffer is updated, not the shader program.

### Top-Activating Examples Explorer

For each neuron, display the top-10 text snippets that maximally activate it. Implementation: `NeuronRecord` stores `most_positive_activation_records` — the highest-activating examples for each neuron. These are pre-computed during a calibration pass over a corpus.

**Backend**:
```
GET /api/ffn/neuron/{layer}/{neuron}/examples?n=10
→ Query pre-computed NeuronRecord
→ Return list of {text, tokens, activations_per_token, max_activation_value}
```

**Frontend**: Render as a scrollable Masonry gallery using `react-virtuoso` for virtualized scrolling. Each card shows the text with tokens color-highlighted by activation intensity, plus the max activation value.

### Neuron "Planet" 3D Map

**Implementation Stack**:

| Component | Technology |
|-----------|-----------|
| **Dimensionality Reduction** | `umap-learn` 0.5+ (Python) with `n_components=3`; fitted once per model on activation matrix `[n_neurons, n_tokens]` |
| **3D Rendering** | `@react-three/fiber` + `@react-three/drei` (Three.js React bindings) |
| **Point Cloud** | `three.js` `InstancedMesh` with `SphereGeometry` — single draw call for all neurons |
| **Labels** | `drei` `Text` component using `troika-three-text` for SDF (signed distance field) text rendering |
| **Post-Processing** | `@react-three/postprocessing` for bloom (glow effect on high-activation neurons) |

**Pipeline**:
```
1. Python: collect activation vectors for all neurons → matrix [N_neurons, D_tokens]
2. Python: fit UMAP(n_components=3) → 3D coordinates for each neuron
3. Store coordinates in SQLite/PostgreSQL (updated only when model changes)
4. Frontend: fetch coordinates + metadata (neuron_index, layer, specialty_label)
5. Three.js: InstancedMesh with 1 sphere per neuron, colored by cluster, sized by avg_activation
6. Interaction: Raycaster for hover/click → show neuron detail panel
```

The AXON project demonstrates this exact pattern: "3D force graph where nodes = SAE features, node size & brightness = activation magnitude".

---

## 📐 ZONE 2: MATH PANEL — Live Formula Rendering

### Implementation Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **LaTeX Rendering** | `KaTeX` 0.16+ (client-side) | 5-10× faster than MathJax; renders to DOM not SVG |
| **Animation Engine** | `GSAP` (GreenSock) + `Framer Motion` | GSAP for math step animations; Framer Motion for layout transitions |
| **Numerical Computation** | `Pyodide` (Python in WebAssembly) | Runs NumPy in browser for live attention computation on small matrices |
| **Formula Typesetting** | Custom React component wrapping KaTeX | Renders formula from JSON-serializable AST (not raw LaTeX strings) |

### Implementation Details

The FFN formula panel shows:

```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
```

Each term is clickable/hoverable, triggering animations that expand the computation:

1. **W₁ · x**: Shows W₁ as a matrix, x as a column vector, and the resulting intermediate vector
2. **b₁ addition**: Shows bias being added
3. **GELU**: Shows the GELU curve with a red dot tracking the current value
4. **W₂ · GELU(...)**: Shows the final projection

**Architecture**:
```
FormulaAST (JSON)
  ├── term: "W_2"
  ├── term: "GELU"
  │   ├── term: "W_1"
  │   ├── term: "x"
  │   └── term: "b_1"
  └── term: "b_2"
```

Each AST node maps to a React component that renders KaTeX and accepts `onHover`, `onClick`, `highlighted` props. When a user hovers over "W₁" in the formula, the corresponding weight matrix lights up in the visualization canvas, and the numerical panel shows the actual values.

### SVD of W₁ — Feature Direction Explorer

**Backend**:
```python
# Compute once when model loads
U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
# Store top-K singular vectors (K=256)
# Each right singular vector Vh[i] is a "feature direction" in the input space
```

**Frontend**: Display the top singular vectors as interactive bar charts. Each bar represents a dimension's contribution. Click a singular vector to see which neurons are most aligned with it (via dot product). The LM Transparency Tool provides exactly this: `decomposed_ffn_out(batch_i, layer, pos)` showing "the decomposed contribution of each neuron in the feed-forward network".

---

## 📈 ZONE 3: ANALYTICS — Diagnostic Dashboard

### Neuron Importance Ranking

**Computation** (Python backend, pre-computed):
```python
# For each neuron i in layer l:
importance_score[i] = mean_activation[i]  # Average activation across corpus
selectivity[i] = 1 - entropy(normalized_activation_distribution[i])  # How selective
dead_score[i] = (max_activation[i] < epsilon)  # Binary: is this neuron dead?
```

NeuroX provides built-in methods for neuron importance extraction: "The toolkit can extract either a local ranking of neurons important to a particular target class, or a global ranking of neurons important to all the target classes".

**Frontend**: Render as a sortable table using `@tanstack/react-table` with virtualized rows. Columns: Rank, Layer, Neuron Index, Importance Score, Selectivity, Status (Active/Dead). Click a row → scroll to that neuron in the heatmap.

### Dead Neuron Detector

Flag neurons where `max_activation < 1e-6` across the entire calibration corpus. Display count per layer as a "health bar" (green → yellow → red). The Neuron Record stores statistical moments (mean, variance, skewness, kurtosis) that enable systematic detection of inactive neurons.

### Neuron Co-Activation Graph

**Computation**:
```python
# For each pair of neurons (i, j):
co_activation[i][j] = correlation(activation_i, activation_j)  # Pearson r
# Threshold at 0.7 → create edge list
```

**Frontend**: Render as an interactive force-directed graph using `deck.gl` `GraphLayer` with GPU-powered layout computation. Nodes = neurons, edges = co-activation relationships. Node color = layer, node size = importance score. Double-click a node to "expand" — show its top-10 activating examples and connected neurons.

---

## 🔬 ZONE 4: INTERPRETABILITY TOOLS

### Neuron Ablation Laboratory

**Implementation**: Uses TransformerLens hook system to zero out specific neurons during forward pass. The hook system "enables the inspection and/or alteration of activations in individual components like attention heads and MLP layers".

**Backend Pipeline**:
```
POST /api/ffn/ablate
Body: { model_id, text, ablated_neurons: [{layer, neuron_index}] }
→ TransformerLens runs forward pass with ablation hooks
→ Returns { original_logits, ablated_logits, logit_diff, affected_tokens }
```

**Frontend**: Toggle switches per neuron (rendered as a grid with `react-window` for performance). Each toggle triggers an API call; results stream back via WebSocket for real-time feedback. Show a "diff view" highlighting which output tokens changed probability.

### SAE (Sparse Autoencoder) Feature View

**Implementation Stack**: `SAELens` library (maintained and actively developed) for training and loading SAEs. SAELens "exists to help researchers train sparse autoencoders and analyse sparse autoencoders" and integrates with `sae-vis` for visualization.

**Pipeline**:
```
1. Train SAE on MLP activations using SAELens (offline, GPU)
2. Store encoder/decoder weights + feature labels
3. Frontend: for each token, run SAE encode → sparse feature vector
4. Display top-K activated features per token as a "feature cloud"
```

The SAE visualizer provides two complementary views: "a feature-centric view (which sequences in a large dataset this feature fires strongest on) and a prompt-centric view (which features score highest on a custom prompt)".

### Neuron Labeling System

Community-contributed labels (e.g., "this neuron detects French text") stored in a dedicated model:

```python
class NeuronLabel(models.Model):
    model_id = models.CharField(max_length=200)
    layer_index = models.IntegerField()
    neuron_index = models.IntegerField()
    label = models.CharField(max_length=500)
    confidence = models.FloatField()  # 0-1, from voting
    proposed_by = models.ForeignKey(User, on_delete=models.CASCADE)
    votes_up = models.IntegerField(default=0)
    votes_down = models.IntegerField(default=0)
    evidence = models.JSONField(default=dict)  # links to top-activating examples
```

This integrates with your existing `community` voting infrastructure and `gpu_reputation` scoring. Neuronpedia provides the model: "EXPLANATION TYPE np_max-act — Forces concise explanations and shows the model the top activating tokens and texts".

---

## 💬 ZONE 5: SOCIAL & COMMUNITY

### "Neuron of the Week"

A scheduled Celery task that selects the most interesting neuron based on: (a) community votes for "interesting" reactions, (b) uniqueness of activation pattern (low similarity to other neurons), (c) quality of community labels. The selected neuron appears on the homepage spotlight, in the weekly digest email, and gets a dedicated discussion thread in the `community` Q&A app.

### Neuron Labeling Game

A gamified interface where users are shown a neuron's top-10 activating examples and asked to propose a label. Labels are voted on by the community. Correct labels (those that survive community voting with high confidence) earn XP and badges. Incorrect labels lose XP. This follows the Yupp model of "token-incentivized feedback" where "diverse user feedback flows freely under blockchain and cryptocurrency economic incentives."

**Architecture**:
```
Frontend: NeuronLabelingGame component
  → Shows neuron metadata + activating examples
  → Text input for label proposal
  → Submit → POST /api/ffn/neuron/{id}/label
Backend: Django view
  → Creates NeuronLabel with proposed_by and confidence=0
  → Emits WebSocket event for real-time leaderboard update
Community: Voting system (up/down) using existing Vote model
```

---

## 🎮 ZONE 6: GAMIFICATION — "Neuron Hunter"

### Game Mechanics

A daily challenge where users are given a description (e.g., "Find the neuron that activates most strongly on question marks") and must navigate the Neuron Activation Universe to locate it. The game uses:

1. **Search mode**: Users can filter neurons by activation statistics (top-activating tokens, activation distribution metrics)
2. **Hints**: Progressively revealed clues (first hint: layer range, second hint: activation pattern description, third hint: specific example)
3. **Scoring**: Points based on time-to-find and number of hints used
4. **Leaderboard**: Daily/weekly/all-time rankings connected to your existing leaderboard infrastructure

**Implementation**:
```
Backend:
  - DailyNeuronChallenge model (date, neuron_id, description, hints[])
  - Pre-compute challenge neuron each day (random selection from "interesting" neurons)
  - POST /api/ffn/challenge/check → verify if user found the correct neuron

Frontend:
  - NeuronHunterGame component with timer, hint system, search interface
  - Connects to Neuron Activation Heatmap (click neuron to "submit guess")
  - Score animation and XP reward on success
```

---

## 🔧 TECHNOLOGY STACK SUMMARY

| Tier | Technology | Purpose |
|------|-----------|---------|
| **Model Inference** | `TransformerLens` 3.0+ with `TransformerBridge` | Hook into 50+ model architectures, extract MLP activations |
| **SAE Training** | `SAELens` + `sae-vis` | Sparse autoencoder training and feature visualization |
| **Neuron Analysis** | `NeuroX` toolkit | Activation extraction, probe training, clustering, ablation |
| **Backend Server** | FastAPI + `asyncio` | REST + WebSocket endpoints for activation data |
| **Data Transfer** | Apache Arrow (`pyarrow`) | Zero-copy columnar format for large activation matrices |
| **Dimensionality Reduction** | `umap-learn` 0.5+ | 3D projection of neuron activation space |
| **Frontend Framework** | React 18 + TypeScript | Component architecture, state management |
| **State Management** | `@tanstack/react-query` + `zustand` | Server state caching + client state |
| **GPU Rendering** | `deck.gl` 9.0 (WebGL2/WebGPU) | GPU-powered heatmap, graph, and scatter layers |
| **3D Rendering** | `@react-three/fiber` + Three.js | Neuron planet map, 3D topology |
| **Math Rendering** | `KaTeX` 0.16+ | Client-side LaTeX formula rendering |
| **Animation** | GSAP + Framer Motion | Step-by-step math animations, transitions |
| **In-Browser Compute** | Pyodide (Python in WebAssembly) | Live attention/FFN computation on small matrices |
| **Virtual Scrolling** | `react-virtuoso` | Efficient rendering of large neuron lists |
| **WebSocket** | Django Channels (existing) | Real-time ablation feedback, collaborative features |

---

## 🗄️ DATA STORAGE & CACHING

| Data | Storage | Retention | Size Estimate |
|------|---------|-----------|---------------|
| Raw activations (per text) | Redis (temporary) | 1 hour | ~50MB per text (12 layers × 3072 neurons × 512 tokens) |
| Neuron statistics | PostgreSQL | Permanent | ~10MB per model |
| Top-activating examples | PostgreSQL (JSONField) | Permanent | ~5MB per model |
| UMAP coordinates | PostgreSQL (3× float) | Until model changes | ~200KB per model |
| SAE features | PostgreSQL + file storage | Permanent | ~500MB per model |
| Neuron labels | PostgreSQL | Permanent | ~1MB per model |

---

## 📏 PERFORMANCE TARGETS

| Operation | Target Latency | Approach |
|-----------|---------------|----------|
| Activation extraction (single text, 512 tokens) | < 2 seconds | GPU inference with cached model |
| Heatmap render (16M cells) | 60fps | GPU instanced rendering via deck.gl |
| 3D neuron map (10K neurons) | 60fps | InstancedMesh with LOD |
| Neuron ablation (single neuron) | < 500ms | Single forward pass with hook |
| Top-examples query | < 100ms | Pre-computed, indexed by neuron_id |
| SAE feature extraction (per token) | < 50ms | Pre-loaded encoder weights |

---

## 🚀 IMPLEMENTATION PHASES

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1: Core Activation Pipeline** | Weeks 1-3 | TransformerLens integration, FastAPI server, Arrow data transfer, basic heatmap |
| **Phase 2: Neuron Analytics** | Weeks 4-5 | Importance ranking, dead neuron detection, co-activation graph, top-examples explorer |
| **Phase 3: Advanced Visualization** | Weeks 6-7 | 3D neuron planet map (UMAP + Three.js), math panel with KaTeX + Pyodide |
| **Phase 4: Interpretability Tools** | Weeks 8-9 | Ablation laboratory, SAE feature view, neuron labeling system |
| **Phase 5: Social & Gamification** | Weeks 10-11 | Neuron of the Week, labeling game, Neuron Hunter game, community voting |
| **Phase 6: Polish & Performance** | Weeks 12-14 | GPU rendering optimization, caching, load testing, documentation |

---

This provides a complete, production-grade implementation blueprint for the FFN/MLP Explorer with every zone detailed in terms of specific technologies, data flow architectures, performance targets, and integration points with your existing Django + React platform.