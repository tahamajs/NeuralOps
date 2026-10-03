# 3D Neural Architecture Visualizer · System Architecture

## 1. System Overview

The 3D Neural Architecture Visualizer is an educational and empirical exploration platform that renders transformer architectures and parameter compression mechanisms in 3D WebGL space.

```
┌────────────────────────────────────────────────────────────────────────┐
│ User Browser                                                           │
│ ┌──────────────────────────────────────────────────────────────────┐   │
│ │ index.html (HTML5, KaTeX CSS/JS, Three.js Import Map)           │   │
│ │ ↓                                                                │   │
│ │ app.js (Bootstrap Entry Point)                                   │   │
│ │ ↓                                                                │   │
│ │ src/index.js (Application Coordinator)                           │   │
│ │ ├── src/core/ (Renderer, Scene, Camera, Lighting, Materials)     │   │
│ │ ├── src/data/ (Spatial Loader, Module Catalog Loader)            │   │
│ │ ├── src/builders/ (Granular Nodes, Macro Blocks, Conduits, Badges)│   │
│ │ ├── src/physics/ (Token Trajectories, Motion Trails, Pulses)     │   │
│ │ ├── src/ui/ (HUD, Inspector, Tree Explorer, Module Modal)        │   │
│ │ ├── src/educational/ (Explanations, LaTeX, Lifecycles, Pipelines)│   │
│ │ └── src/state/ (Observable AppStore)                             │   │
│ └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory & Module Structure

```
Docs/visualizer/
├── index.html                  # Accessible layout, HUD cards, modal shells, KaTeX
├── app.js                      # Modular bootstrap (< 30 lines)
├── style.css                   # Cybernetic dark theme styles
├── standalone.html             # Offline single-file bundle
├── spatial-data.json           # 3D spatial node coordinates across 4 models
├── modules-catalog.json        # Index of all 240+ research modules
├── modules_components.json     # Consolidated sublayer component topologies
├── components/                 # 240 individual component JSON files
├── docs/                       # Architectural & tutorial documentation
│   ├── ARCHITECTURE.md
│   ├── DATA_SCHEMA.md
│   ├── TOKEN_PHYSICS.md
│   ├── LOD_SYSTEM.md
│   ├── EXTENDING.md
│   ├── CONDUIT_SYSTEM.md
│   ├── VERIFICATION_REPORT.md
│   └── TUTORIALS/
│       ├── 01_setup.md
│       ├── 02_add_module.md
│       ├── 03_add_model.md
│       └── 04_custom_shaders.md
├── src/                        # Modular source code
│   ├── constants.js            # All magic numbers, color palettes, and model specs
│   ├── index.js                # Main application coordinator
│   ├── types/                  # TypeScript interface contracts
│   ├── core/                   # WebGL renderer, scene, camera, materials
│   ├── data/                   # JSON data loaders
│   ├── builders/               # Mesh and conduit scene builders
│   ├── physics/                # Token trajectory and pulse animations
│   ├── interactions/           # Camera transitions, raycasting, selection
│   ├── ui/                     # Inspector drawer, tree explorer, module catalog
│   ├── educational/            # Math rendering, lifecycles, and pedagogical notes
│   ├── state/                  # Observable state store
│   └── utils/                  # Formatters, DOM, and math helpers
└── tests/                      # Automated unit test suite
    ├── constants.test.js
    ├── token_trajectory.test.js
    └── schema_validation.test.js
```

---

## 3. Key Subsystems

### 1. Dual-Viewport Comparison
The scene renders two vertical computational columns side-by-side:
* **Baseline Column (Left, $X = -4.5$)**: The uncompressed dense monolithic architecture ($O(N^2)$ quadratic attention, dense MLP projections).
* **Compressed Column (Right, $X = +4.5$)**: The active compression paradigm (SVD rank bottleneck, Mixture-of-Experts routing, DeepSeek MLA latent KV caching, or SliceGPT channel slicing).

### 2. Semantic Zoom & Continuous Level-of-Detail (LOD)
Visual density dynamically adapts to camera proximity:
* **Far View (LOD 0, Distance > 40)**: Layers collapse into consolidated enclosing macro bounding boxes.
* **Transition Zone (Distance 25–40)**: Macro boxes smoothly fade as internal matrix tensors materialize.
* **Near View (LOD 1, Distance < 25)**: Full granular tensor boxes, attention head partitions, SVD necks, and floating dimension badges (`[768 × 768]`, `12H`, `r=64`).

### 3. Code-Accurate Token Dynamics
Particles are not decorative; their 3D paths evaluate real transformer dataflow equations:
* **Qwen 2.5/3.5**: Multi-query fan-out with 14 Query heads, 2 shared Key/Value head bundles, and SwiGLU `gate_proj`/`up_proj` bifurcations.
* **GPT-2**: Unified travel through fused $c\_attn$ Conv1D wide tensor projections.
* **DistilBERT / RoBERTa**: Physical constriction into the rank-64 SVD Factor $V$ eigenspace, expanding through Factor $U$.
* **Residual Highways**: 25% of tokens bypass sublayers to visualize the identity skip addition: $X_{\text{out}} = X_{\text{in}} + \text{Sublayer}(\text{Norm}(X_{\text{in}}))$.
