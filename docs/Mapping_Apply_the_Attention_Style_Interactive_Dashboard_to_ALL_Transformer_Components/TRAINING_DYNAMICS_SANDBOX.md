# Zone 11: Training Dynamics Sandbox — Complete Implementation Specifications

This document provides a **fully exhaustive implementation blueprint** for adding a real‑time, interactive training dynamics sandbox to the NeuralOps Attention Dashboard. Every sub‑component is specified with its programming language, framework, library dependencies, data flow, rendering approach, and integration points.

---

## 11.1 Architecture Overview

The Training Dynamics Sandbox consists of **five interconnected rendering surfaces** that share a common training-state data source and are synchronized through a unified time‑slider controller. Data flows from the PyTorch training loop through a WebSocket bridge into the browser, where React state drives three.js/WebGL, SVG, and Canvas visualizations simultaneously.

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING DYNAMICS SANDBOX                      │
│                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │  CANVAS 1          │  │  CANVAS 2          │                  │
│  │  2D Embedding      │  │  Loss Landscape    │                  │
│  │  Sandbox           │  │  Explorer          │                  │
│  │  (Three.js/WebGL)  │  │  (Plotly.js+WebGL) │                  │
│  └───────────────────┘  └───────────────────┘                   │
│                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐                   │
│  │  CANVAS 3          │  │  MATH PANEL        │                  │
│  │  Gradient Flow     │  │  Chain Rule        │                  │
│  │  Animation         │  │  Step-by-Step      │                  │
│  │  (D3.js + SVG)     │  │  (KaTeX + CSS)     │                  │
│  └───────────────────┘  └───────────────────┘                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  ANALYTICS PANEL (Recharts + SVG)                            │ │
│  │  Gradient Norm Ratio | Update/Weight Ratio | Loss Curves     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  CONTROL BAR (React + Tailwind)                              │ │
│  │  [▶/⏸] [Step Slider ████████░░░░] [LR ▾] [Optimizer ▾] [BS ▾]│ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11.2 Canvas 1: 2D Embedding Training Sandbox — `EmbeddingSandbox.tsx`

> *Inspiration: Visual Sandbox (Stanford 2024) constrains transformer embeddings to d=2, allowing each layer's dynamics to be plotted as particles moving in a 2D plane.*

### 11.2.1 Core Concept

For a small‑scale transformer where **d_model = 2**, every token embedding is a 2D point. During training, the weight matrices **W_Q**, **W_K**, **W_V**, and **W_O** evolve, and the effect is visible as tokens physically **moving** in the 2D plane. The Visual Sandbox paper "reduces the problem to the study of a system of particles, which interact according to properties of the data distribution and correlations between embeddings, and shows that training dynamics can be understood through particle interactions".

### 11.2.2 Implementation Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | TypeScript (strict mode) | Type safety for complex geometry |
| **3D Rendering** | **Three.js** (v0.170+) via `@react-three/fiber` | WebGL‑accelerated 2D particle rendering disguised as 3D scene (orthographic camera) |
| **Particle System** | `three.js` `InstancedMesh` + custom `ShaderMaterial` | GPU‑accelerated rendering of 1,000+ token particles with per‑particle color, size, and opacity |
| **Animation Loop** | `useFrame` from `@react-three/fiber` | 60fps render loop synced with training step updates |
| **State Management** | Zustand store (`useTrainingSandboxStore`) | Central store for training state: embeddings per layer per step, loss, gradients |
| **Data Serialization** | Protocol Buffers (protobuf) via `protobufjs` | Efficient binary serialization of embedding tensors from Python to browser |
| **WebSocket Transport** | Django Channels (existing) + custom binary WebSocket sub‑protocol | Streaming embedding snapshots at each training step |

### 11.2.3 Data Flow

```
[PyTorch Training Loop]
    │
    ├── On each optimizer step:
    │   ├── Capture embeddings: layer.attention.q_proj(input) → 2D points
    │   ├── Serialize via protobuf: { step, layer, token_embeddings: [{x, y, token_id}] }
    │   └── Send via WebSocket binary frame
    │
    ▼
[Django Channels Consumer]
    │  (existing WebSocket infrastructure)
    ├── Deserialize protobuf → JSON
    └── Broadcast to channel group "training_sandbox_{run_id}"
    │
    ▼
[React WebSocket Hook: useTrainingStream]
    ├── Receives JSON snapshots
    ├── Updates Zustand store: trainingSnapshots[step] = { layer_embeddings, loss, gradients }
    └── Triggers re-render of Three.js scene
```

### 11.2.4 Rendering Approach

**Scene Setup:**
- Orthographic camera (not perspective) → true 2D view with zoom/pan
- Coordinate system: X axis = embedding dimension 0, Y axis = embedding dimension 1
- Background: dark grid (`#0f172a` with subtle grid lines at intervals)

**Particle Representation:**
- Each token embedding rendered as a `InstancedMesh` sphere
- Per‑instance attributes: `color` (by token position), `size` (by attention weight), `opacity`
- Custom GLSL vertex shader applies Gaussian blur for soft glow
- Custom GLSL fragment shader computes Fresnel‑like edge glow

**Animation Between Steps:**
- Use `requestAnimationFrame` with linear interpolation (lerp) between current and target positions
- Spring physics via `@react-spring/three` for smooth transitions when user scrubs the timeline
- Motion trails: each particle leaves a fading trail (stored as a `Line` geometry with decaying opacity)

**Token Connections (Attention):**
- For the currently selected layer/head, draw semi‑transparent lines between tokens with attention weight > 0.1
- Line opacity = attention weight; line color = head identity
- Lines rendered as `LineSegments` with `LineBasicMaterial`

**Layer Switching:**
- User clicks "Layer 0" → "Layer 5" → all particles smoothly animate to their new positions
- Layer indicator shows: `Layer {n}: Q/K/V/O transformations applied`

### 11.2.5 Particle Interaction Model (Research Insight)

The Visual Sandbox paper shows that embedding vectors "tend to organize into two types of structures: grids and circles" during training. The implementation should:
- Auto‑detect and highlight these emergent structures (grid detection via Hough transform, circle detection via RANSAC)
- Show "particle interaction forces" as vector arrows between tokens (computed from attention weights)
- Display the "circuit formation" as tokens that consistently attend to each other form visible clusters

---

## 11.3 Canvas 2: Loss Landscape Explorer — `LossLandscapeExplorer.tsx`

> *Inspiration: `loss-landscapes` library, `Landscaper` (2025, Vis4SciML), `GradVis` toolbox.*

### 11.3.1 Core Concept

The loss landscape is a 3D surface where X and Y axes represent random directions in parameter space (using filter normalization), and the Z axis (height/color) represents the loss value. The optimization trajectory is plotted as a path on this surface, showing how the optimizer navigates the loss terrain.

### 11.3.2 Implementation Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Loss Landscape Computation** | `loss-landscapes` PyTorch library (v0.5+) | Computes loss values along random parameter directions with filter normalization |
| **Topological Analysis** | `Landscaper` (v1.0+) | Hessian‑based subspace construction and topological data analysis for basin detection |
| **3D Surface Rendering** | **Plotly.js** (`react-plotly.js`) with `gl3d` trace type | High‑quality 3D surface plots with built‑in camera controls, lighting, and contour lines |
| **Trajectory Overlay** | Plotly `scatter3d` trace | Optimization path as a colored 3D line (color gradient from blue=start to red=end) |
| **Real‑time Updates** | Plotly `extendTraces` / `restyle` | Incremental surface updates without full re‑render |
| **Python Backend** | Celery task | Offloads expensive loss landscape computation to worker nodes |
| **Caching** | Redis | Caches computed landscape data for replay without recomputation |

### 11.3.3 Computation Pipeline (Python Side)

```python
# gpu_benchmark/loss_landscape_task.py (conceptual)
import loss_landscapes
import landscaper
from loss_landscapes.model_interface import ModelInterface

@celery_app.task(bind=True)
def compute_loss_landscape(self, model_state_dict, data_batch, resolution=50):
    # 1. Restore model
    model = build_model_from_state(model_state_dict)
    metric = loss_landscapes.metrics.Loss()

    # 2. Compute random plane with filter normalization
    landscape = loss_landscapes.random_plane(
        model, metric,
        distance=5, steps=resolution,
        normalization='filter',  # Li et al. 2018 normalization
    )

    # 3. (Optional) Advanced: Landscaper Hessian subspace
    # hessian_subspace = landscaper.HessianSubspace(model, data_batch)
    # landscape = landscaper.sample_landscape(hessian_subspace, resolution)

    # 4. Return as JSON for Plotly
    return {
        'x': landscape.coords[0].tolist(),  # alpha values
        'y': landscape.coords[1].tolist(),  # beta values
        'z': landscape.loss_values.tolist(),
    }
```

### 11.3.4 Rendering Approach

**Plotly Configuration:**
```typescript
const layout = {
  scene: {
    xaxis: { title: 'Direction α (filter-normalized)', color: '#94a3b8' },
    yaxis: { title: 'Direction β (filter-normalized)', color: '#94a3b8' },
    zaxis: { title: 'Loss', color: '#94a3b8' },
    camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
  },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
}
```

**Surface Trace:**
- Type: `surface`
- Colorscale: `'RdYlBu_r'` (red=high loss, blue=low loss)
- Contours: projected on XZ and YZ planes
- Opacity: 0.85

**Trajectory Trace:**
- Type: `scatter3d`
- Mode: `'lines+markers'`
- Line color gradient: `[[0, '#3b82f6'], [1, '#ef4444']]` (blue to red as training progresses)
- Marker at current position: large glowing sphere (via `marker.symbol: 'circle'`, `marker.size: 8`)

**Interactive Features:**
- **Click on trajectory point** → jumps training step slider to that position
- **Hover on surface** → shows exact loss value
- **Toggle contour lines** on/off
- **Export** as PNG/SVG for papers

### 11.3.5 Filter Normalization Detail

The filter normalization technique (Li et al. 2018, "Visualizing the Loss Landscape of Neural Nets") is critical for producing meaningful loss landscapes for large models. Without it, the landscape appears flat because parameter scales vary across layers. The implementation should:

1. For each random direction vector **d**, normalize each filter (convolutional kernel or weight matrix row) to have the same Frobenius norm as the corresponding filter in the original parameters
2. This ensures that all layers contribute proportionally to the perturbation, revealing the true curvature of the loss surface

---

## 11.4 Canvas 3: Gradient Flow Animation — `GradientFlowAnimation.tsx`

> *Inspiration: `gradviz` PyPI package, `RNNbow` (IEEE CG&A 2018), `TorchExplorer`.*

### 11.4.1 Core Concept

Gradient flow visualization shows how gradient magnitudes propagate backward through the network during backpropagation. Each layer is represented as a vertical bar whose height/color corresponds to the L2 norm of its gradients. The visualization uses RNNbow's approach of "displaying the relative gradient contributions from cells in a neighborhood" but generalized to transformer layers.

### 11.4.2 Implementation Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Gradient Capture** | PyTorch hook: `module.register_full_backward_hook()` | Captures gradient norms per layer during backpropagation |
| **Gradient Storage** | `gradviz` (PyPI, 2025) | Saves gradient L2 norms to CSV during training |
| **Chart Rendering** | **D3.js** (v7) with React wrapper (`react-d3`) | Custom animated bar chart with gradient flow particles |
| **Particle Animation** | Custom Canvas2D particle system | Animated gradient "packets" flowing backward through layer bars |
| **Color Encoding** | `d3-scale-chromatic` with `interpolateRdYlBu` | Color gradient from blue (small gradient) to red (large gradient / potential explosion) |

### 11.4.3 Gradient Capture (Python Side)

```python
# gpu_benchmark/gradient_capture.py (conceptual)
import torch

class GradientCaptureHook:
    """Captures gradient norms for all nn.Linear modules during training."""
    def __init__(self, model):
        self.gradient_norms = {}
        self.hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_full_backward_hook(
                    lambda m, g_in, g_out, n=name: self._capture(n, g_in, g_out)
                )
                self.hooks.append(hook)

    def _capture(self, name, grad_input, grad_output):
        # grad_input[0] is gradient w.r.t. module input
        # grad_output[0] is gradient w.r.t. module output
        self.gradient_norms[name] = {
            'input_grad_norm': grad_input[0].norm().item(),
            'weight_grad_norm': module.weight.grad.norm().item() if module.weight.grad is not None else 0,
            'output_grad_norm': grad_output[0].norm().item(),
        }

    def snapshot(self):
        return self.gradient_norms.copy()
```

### 11.4.4 Rendering Approach

**Layer Bar Chart:**
- Each layer rendered as a horizontal bar (D3 `<rect>`)
- Bar height proportional to gradient L2 norm (log scale for visibility)
- Bar color mapped via `d3.interpolateRdYlBu`: blue = small gradient, red = large gradient
- Layer labels on the y‑axis: "Embedding", "Layer 0 Attn Q", "Layer 0 Attn K", ..., "Layer N FFN Out"
- Threshold lines: horizontal dashed lines at gradient norm = 1.0 (healthy) and gradient norm = 100 (exploding)

**Gradient Flow Particles:**
- Animated particles (Canvas2D overlay) flow from right (output) to left (input)
- Particle speed = gradient magnitude (faster for larger gradients)
- Particle density = number of parameters in that layer
- Particles burst into red sparkles when gradient norm exceeds explosion threshold

**Animation Control:**
- Play/Pause button for the gradient flow animation
- Speed control slider (0.5× to 4×)
- Step forward/backward one training iteration

**Comparison Mode (RNNbow‑inspired):**
- Show gradient norms for two different training runs side‑by‑side
- Highlight layers where gradient behavior diverges significantly
- Useful for comparing optimizer configurations or learning rates

---

## 11.5 Math Panel: Backpropagation Chain Rule — `ChainRuleMathPanel.tsx`

### 11.5.1 Core Concept

When the user clicks any layer in the gradient flow visualization, the math panel shows the **exact chain rule computation** for that layer's weight gradient, with live numerical values substituted from the actual training data.

### 11.5.2 Implementation Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LaTeX Rendering** | **KaTeX** (v0.16+) via `react-katex` | Fast, server‑side renderable LaTeX for mathematical expressions |
| **Step Animation** | CSS `@keyframes` + React state machine | Animated highlighting of each computation step |
| **Numerical Values** | React state from WebSocket | Live substitution of actual gradient values from training |
| **Diagram** | D3.js force‑directed graph | Visual representation of the computation graph for the selected layer |

### 11.5.3 Formula Display

For a linear layer `y = xW^T + b`, the chain rule is:

```
∂L/∂W = ∂L/∂y · ∂y/∂W = (∂L/∂y)^T · x
```

The panel shows:
1. **Step 1**: `∂L/∂y` — the upstream gradient (displayed as a matrix of actual values)
2. **Step 2**: `x` — the layer input (displayed as a matrix)
3. **Step 3**: The multiplication (animated as matrices sliding together)
4. **Result**: `∂L/∂W` — the weight gradient (displayed with color‑coded magnitude)

For the **attention mechanism**, the chain rule is more complex:

```
∂L/∂W_Q = ∂L/∂A · ∂A/∂S · ∂S/∂Q · ∂Q/∂W_Q
```

where:
- `∂L/∂A` = gradient of loss w.r.t. attention weights
- `∂A/∂S` = Jacobian of softmax
- `∂S/∂Q` = K^T / √d_k
- `∂Q/∂W_Q` = x^T (input)

Each sub‑step is expandable/collapsible, with numerical values populated in real time.

### 11.5.4 Interactive Features

- **Hover** on any variable → highlights its source in the gradient flow chart and embedding sandbox
- **Click** on a variable → "pin" it, showing its value across all training steps as a mini line chart
- **"Explain This" button** → triggers an LLM‑powered natural language explanation of the chain rule step (via your existing agent infrastructure)
- **Export to ManimML** → generates a Manim animation script for creating educational videos

---

## 11.6 Analytics Panel — `TrainingAnalyticsPanel.tsx`

### 11.6.1 Metrics Displayed

| Metric | Computation | Visualization | Diagnostic Use |
|--------|------------|---------------|----------------|
| **Gradient Norm Ratio** | `‖grad_L_n‖ / ‖grad_L_1‖` for each layer L_n | Grouped bar chart (Recharts) | Detects vanishing gradients (< 0.01) or exploding gradients (> 100) |
| **Update‑to‑Weight Ratio** | `‖η · grad‖ / ‖W‖` for each parameter | Scatter plot (D3) | Values > 0.01 indicate rapid learning; < 1e-6 indicate stagnation |
| **Loss Curve Overlay** | Training loss + validation loss per step | Line chart (Recharts) with `referenceLine` at key events | Overlay multiple runs for comparison |
| **Gradient Covariance** | Condition number of gradient covariance matrix | Heatmap (custom SVG) | High condition number → ill‑conditioned optimization |
| **Weight Distribution Evolution** | Histogram of weight values per layer per epoch | Animated histogram (D3 + `transition()`) | Detects mode collapse or weight saturation |

### 11.6.2 Implementation

- **Recharts** (`react-recharts`) for all line/bar charts — lightweight, React‑native, supports animations and reference lines
- **D3.js** for custom scatter plots and heatmaps where Recharts is insufficient
- **Data Source**: WebSocket stream from training loop (same as gradient capture)
- **Caching**: IndexedDB (via `idb` library) stores historical metrics for replay without server round‑trips

---

## 11.7 Control Bar — `TrainingControls.tsx`

### 11.7.1 Components

| Control | Implementation | Behavior |
|---------|---------------|----------|
| **Training Step Slider** | `<input type="range">` with custom Tailwind styling | Rewind/fast‑forward through all recorded training steps; all four canvases update simultaneously |
| **Play/Pause** | React state toggle | Auto‑advances the step slider at configurable speed (1 step/sec to 60 steps/sec) |
| **Learning Rate Selector** | Custom `<select>` with preset values + custom input | Changes the LR for a *new* training run (not retroactive) |
| **Optimizer Selector** | Dropdown: SGD, Adam, AdamW, RMSprop | Shows optimizer‑specific hyperparameters when selected |
| **Batch Size Selector** | Dropdown: 8, 16, 32, 64, 128 | Adjusts batch size for new training runs |
| **"Launch Training" Button** | `onClick` → API call to experiment launcher | Opens experiment launcher with all sandbox parameters pre‑filled |

### 11.7.2 Time Synchronization

All canvases subscribe to the same Zustand store key: `trainingSandbox.currentStep`. When the slider moves:
1. The store updates `currentStep`
2. Each canvas's `useEffect` detects the change and updates its visualization
3. For the embedding sandbox, particle positions are interpolated between the nearest recorded snapshots
4. For the loss landscape, the trajectory marker moves to the corresponding position

---

## 11.8 Integration with Existing Platform

### 11.8.1 WebSocket Infrastructure

- Reuses existing Django Channels `TrainingRunConsumer`
- New message type: `training_sandbox_snapshot` containing serialized embeddings, gradients, loss
- Binary protocol via protobuf for efficiency (embedding tensors can be large)
- Fallback to JSON for debugging

### 11.8.2 Database Storage

- New model: `TrainingSandboxSnapshot` (in `experiment/models.py`)
  - `run`: FK to `TrainingRun`
  - `step`: IntegerField
  - `embeddings_json`: JSONField (stores 2D embedding coordinates per layer per token)
  - `gradient_norms_json`: JSONField (stores gradient norms per layer)
  - `loss`: FloatField
- Snapshots are saved every N training steps (configurable, default N=10)
- Enable replay of training dynamics without re‑running training

### 11.8.3 Connection to Existing Dashboard

- The Training Dynamics Sandbox is a **new tab** in the Attention Visualization Dashboard
- It shares the model selector, text input, and layer/head controls with the attention views
- The "Launch Training" button connects to the existing experiment launcher
- Training run results are linked to the knowledge graph

---

## 11.9 Performance Optimization

### 11.9.1 Rendering Performance

- **Three.js InstancedMesh**: renders all token particles in a single draw call (critical for 1000+ tokens)
- **Web Worker**: offloads protobuf deserialization and data processing to a dedicated worker thread
- **RequestAnimationFrame throttling**: limit embedding sandbox to 30fps when tab is not visible (via `Page Visibility API`)
- **Level‑of‑Detail (LOD)**: when zoomed out, tokens are rendered as simple points; when zoomed in, as detailed spheres with glow

### 11.9.2 Data Transfer Optimization

- **Delta encoding**: only transmit changed values between steps (most embeddings change incrementally)
- **Quantization**: reduce float32 → float16 for embedding coordinates (2D, so precision loss is negligible)
- **Chunking**: split large snapshots across multiple WebSocket frames to avoid blocking

---

## 11.10 Summary of All Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend Framework** | React 18 + TypeScript (strict) | Component architecture and type safety |
| **3D Rendering** | Three.js + `@react-three/fiber` + `@react-three/drei` | GPU‑accelerated 2D embedding particle sandbox |
| **Charting** | Recharts, Plotly.js, D3.js v7 | Loss landscape surface, gradient flow charts, analytics |
| **Math Rendering** | KaTeX (`react-katex`) | Live LaTeX equation display |
| **Animation** | `@react-spring/three`, CSS `@keyframes`, D3 transitions | Smooth particle interpolation, math step highlighting |
| **State Management** | Zustand + Immer middleware | Central training state with immutable updates |
| **WebSocket** | Django Channels (existing) + custom protobuf sub‑protocol | Streaming training snapshots |
| **Serialization** | Protocol Buffers (`protobufjs`) | Efficient binary serialization of embedding data |
| **Backend Computation** | `loss-landscapes` PyTorch library, `Landscaper` (2025), `gradviz`, `TorchExplorer` | Loss landscape, gradient capture, topological analysis |
| **Caching** | Redis (server), IndexedDB (`idb`, browser) | Landscape caching, historical metrics |
| **Worker Threads** | Web Worker API | Off‑main‑thread protobuf deserialization |
| **CSS Framework** | Tailwind CSS (existing) | Consistent styling with the rest of the dashboard |

---

## 11.11 Implementation Roadmap

| Week | Deliverable | Key Milestone |
|------|------------|---------------|
| **Week 1** | Gradient capture hooks + WebSocket streaming + Zustand store + Gradient Flow Animation (Canvas 3) | Real‑time gradient bars visible during training |
| **Week 2** | 2D Embedding Sandbox (Canvas 1) with Three.js InstancedMesh + particle interpolation | Tokens visible as moving particles in 2D plane |
| **Week 3** | Loss Landscape Explorer (Canvas 2) with Plotly.js + Python computation backend | 3D loss surface with trajectory overlay |
| **Week 4** | Math Panel (Chain Rule) + Analytics Panel + Control Bar + full integration | Complete sandbox with all five panels synchronized |
| **Week 5** | Performance optimization (protobuf, Web Worker, delta encoding) + testing | 60fps performance with 1000+ tokens |
| **Week 6** | Replay mode (IndexedDB caching) + integration with experiment launcher | Full user workflow: explore → launch training → analyze |

This implementation plan provides a complete, production‑ready training dynamics sandbox that integrates deeply with your existing NeuralOps platform while leveraging the most advanced visualization libraries available in 2025‑2026.