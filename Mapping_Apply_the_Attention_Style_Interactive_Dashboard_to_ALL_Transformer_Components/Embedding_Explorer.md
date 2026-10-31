# 🧬 Embedding Explorer — Complete Implementation Specification

---

## Architecture Overview

The Embedding Explorer is designed as a **browser‑native, zero‑upload application** following the architecture pioneered by Apple's Embedding Atlas — all computation (embedding generation, dimensionality reduction, clustering, similarity search) executes entirely on the client side via WebAssembly, eliminating the need for any backend server beyond your existing Django API. The rendering pipeline uses **WebGPU** as the primary backend with **WebGL2 fallback**, enabling smooth interaction with millions of data points while maintaining 60fps on consumer hardware.

The system is organized into **six interconnected architectural layers** that map to the dashboard zones the user specified:

---

## 🔧 Layer 0: Core Computation Engine — Rust → WebAssembly Pipeline

### Framework: `wasm-pack` + `rustc` → `.wasm` binary

The heart of the embedding explorer is a **Rust codebase compiled to WebAssembly** that handles all heavy computation directly in the browser. This follows Apple's approach of combining "Rust‑based clustering modules with WebAssembly UMAP implementations to significantly optimize dimensionality reduction performance".

**Modules compiled to WASM:**

| Module | Rust Crate | WASM Package | Purpose |
|--------|-----------|--------------|---------|
| Dimensionality Reduction | `umap-rs` | `@neuralops/umap-wasm` | UMAP/t‑SNE/PCA projection of 768‑dim vectors to 2D/3D |
| Vector Similarity | `simsimd` | `@neuralops/simd-wasm` | SIMD‑accelerated cosine similarity, dot product, Euclidean distance on Float32 arrays |
| Clustering Engine | `linfa-clustering` | `@neuralops/cluster-wasm` | HDBSCAN, k‑means, mean‑shift for automatic cluster labeling |
| Linear Classification | `liblinear-rs` wrapper | `@wlearn/liblinear` | Logistic regression / linear SVM for probe training directly in browser |
| KDE Renderer | `kde-rs` | `@neuralops/kde-wasm` | Kernel density estimation for contour/density heatmap generation |

**Communication Pattern:** The WASM modules receive `Float64Array` typed arrays via a shared `ArrayBuffer` to avoid JavaScript↔WASM copy overhead. The `@wlearn/liblinear` package demonstrates this pattern — it exposes an API accepting `{ data: Float64Array, rows, cols }` for zero‑copy training.

**Initialization Sequence (on page load):**
1. Web Worker spawns and loads all WASM modules in parallel via `WebAssembly.instantiateStreaming`
2. Worker exposes a message‑passing API: `{ type: 'umap_project', vectors: Float64Array, params: {...} }`
3. Main thread sends typed arrays via `postMessage` (transferred, not copied, via `transferList`)
4. Worker returns projected coordinates, cluster labels, or similarity scores
5. Results flow into the React state via a Zustand store slice

---

## 🧭 Zone 1: 3D Embedding Space Explorer

### Rendering Framework: `@react-three/fiber` (R3F) + Three.js + WebGPU backend

The 3D canvas uses **React Three Fiber** as the declarative React wrapper around Three.js, with the **WebGPURenderer** (`three@latest` with `WebGPURenderer` from `three/webgpu`) as the primary renderer and `WebGLRenderer` as fallback. This approach mirrors Apple's strategy: "WebGPU is adopted as the default rendering backend (with WebGL 2 compatibility layer)". The Project Golem project validates this architecture — it "uses UMAP to project high‑dimensional embeddings into 3D and Three.js to render them interactively in the browser".

### Component Architecture:

```
<EmbeddingSpace3D>
  ├── <WebGPURenderer fallback={<WebGLRenderer />} />
  ├── <OrbitControls />                          ← camera rotation/zoom/pan
  ├── <TokenSphereCloud>                         ← primary rendering
  │   ├── <InstancedMesh args={[sphereGeometry, material, count]}>
  │   │   ← GPU‑instanced rendering for millions of spheres
  │   └── <HoverHighlight />                     ← raycasting on hover
  ├── <SemanticClusters>                         ← cluster envelopes
  │   └── <ConvexHullMesh />                     ← transparent hull per cluster
  ├── <TokenTrajectoryLines>                     ← semantic pathway
  │   └── <Line segments />                      ← line from layer N to N+1
  ├── <SearchTeleporter />                       ← "teleport to token"
  ├── <SemanticAxisIndicator />                  ← PCA/ICA axis arrows
  └── <DensityContour />                         ← KDE contour overlay
```

### GPU Instanced Rendering Strategy

For **1 million+ spheres**, standard React rendering (one `<mesh>` per sphere) would crash. Instead, the `TokenSphereCloud` component uses **GPU instancing** via Three.js `InstancedMesh`:

1. **Geometry:** A single `SphereGeometry(radius=0.02, segments=8, rings=8)` is created once
2. **Material:** `MeshStandardMaterial` with per‑instance color via `instanceColor` buffer attribute
3. **Matrix Buffer:** A `Float32Array` of size `count × 16` holds the model matrix for each instance (position from UMAP coords, uniform scale)
4. **Color Buffer:** A `Float32Array` of size `count × 3` holds the color for each token (mapped from cluster label, token type, or attention weight)
5. **Update on hover:** The raycasting system identifies the instance index; the matrix and color for that instance are updated to show a glow/highlight

This is the same technique used by DiVE: "an interactive 3D web viewer of up to million points on one screen ... based on graphosaurus.js and three.js". The Helios‑Web project further validates "GPU‑based rendering for real‑time visualization and interactive analysis of massive‑scale networks and high‑dimensional embeddings" using WebGL.

### Token Search & Teleport

The **Search Teleporter** component provides a text input where users type any word/token. Under the hood:

1. **Frontend:** User types "Paris" → React state updates
2. **WASM Worker:** The token's embedding vector is retrieved from the model's embedding matrix (loaded as a `Float32Array` at page initialization)
3. **Nearest‑neighbor search:** The WASM similarity module finds the 50 nearest tokens in embedding space and returns their indices + distances
4. **Camera animation:** `useFrame` in R3F animates the camera from its current position to the target sphere's position using `THREE.Vector3.lerp` with an easing function
5. **Highlight effect:** The target sphere's emissive intensity is set to 1.0, and its scale is temporarily increased

### Semantic Cluster Rendering

After the WASM clustering module computes cluster assignments, each cluster is rendered as a **transparent convex hull** (or alpha shape). The implementation uses the `quickhull3d` algorithm compiled to WASM or a JavaScript port:

1. Cluster centers and boundaries are computed by the WASM clustering module
2. For each cluster with ≥ 5 points, a `ConvexGeometry` is generated from the outermost points
3. The hull is rendered as a semi‑transparent mesh with cluster‑specific color
4. A **text label** (the cluster's auto‑generated name like "food words" or "location terms") is positioned at the hull centroid using `@react-three/drei`'s `<Text>` component

### Token Trajectory View

Inspired by **Prompt Rover** which "builds a minimum path graph between embeddings and projects it with t‑SNE to trace the semantic trajectory from input to output", and by **InterpreT** which provides "the ability to track and visualize token embeddings through each layer of a Transformer":

1. When a user selects a specific token in a specific sentence, the system queries your Django API for that token's hidden state at **every layer** of the model
2. Each layer's hidden state is projected into the same 3D space (using the same UMAP projection previously computed)
3. The positions are connected by a **curved tube** (using `CatmullRomCurve3` in Three.js) with a color gradient from blue (layer 0) to red (layer 12)
4. The tube thickness varies to represent the **change magnitude** between consecutive layers (where the representation changes most)
5. The tube animates on hover — a glowing particle travels along the path from input to output

---

## 🔬 Zone 2: Embedding Matrix Heatmap

### Rendering: HTML5 Canvas 2D with OffscreenCanvas + Web Worker

For the `vocab_size × d_model` matrix (typically 30,000+ × 768), a canvas‑based approach is required because SVG with that many elements would crash. The implementation:

1. **OffscreenCanvas** in a Web Worker renders the full matrix at high resolution into an `ImageBitmap`
2. The `ImageBitmap` is `transferToImageBitmap()` to the main thread and drawn onto a visible canvas
3. **Zoom/Pan:** The visible canvas shows a viewport into the full matrix. Zooming in triggers re‑rendering at higher resolution for the visible region only
4. **Color scale:** Blue for negative values, white for zero, red for positive (matching the 3Blue1Brown visualization convention)
5. **Search:** Typing a token name scrolls the viewport to that row and highlights it with a yellow border
6. **Row comparison:** Selecting two tokens renders both rows side‑by‑side with a difference row showing the delta

### Dimension Importance via PCA

The WASM PCA module computes:
1. **Explained variance ratio** per principal component
2. The top‑10 dimensions are rendered as a bar chart
3. The "semantic axis discovery" feature (powered by ICA) is described in Zone 5 below

---

## 📐 Zone 3: Live Math Panel — Embedding Formulas

### Rendering: `react-latex` / `katex` for LaTeX, `framer-motion` for animation

The math panel displays the embedding computation formula:

```
E[token] = W_e[token_id, :] + W_p[pos, :]
```

Where:
- `W_e`: The token embedding matrix (learned during training)
- `W_p`: The positional encoding matrix (sinusoidal or learned)

### Animated Computation Steps

Inspired by the Self Attention Flow tool and 3Blue1Brown's Manim animations:

1. **Step 1:** The token embedding row is highlighted in the heatmap, values animate from gray to color
2. **Step 2:** The positional encoding vector is rendered as a separate bar chart
3. **Step 3:** The two vectors "merge" (element‑wise addition animation) to form the final embedding
4. **Step 4:** The final embedding vector is shown, and the user can see which dimensions are dominated by token identity vs. position

### Cosine Similarity Calculator

1. User selects any two tokens from dropdowns
2. The WASM similarity module computes `cos(a,b) = (a·b) / (||a||·||b||)`
3. The result is displayed as a large number (0.0–1.0) with a color‑coded bar
4. **Dimension contribution breakdown:** A bar chart shows which dimensions contribute most to the similarity (the element‑wise product `a_i × b_i` before summation)
5. The computation uses the `wasm-similarity` library which "moves the hot loop into WebAssembly compiled from Rust" for "high-performance vector similarity and distance metrics"

---

## 📈 Zone 4: Embedding Statistics Dashboard

### Vocabulary Coverage Histogram

Rendered using `recharts` (already in your project). The histogram shows how often each token appears in a calibration corpus, revealing the long‑tail distribution typical of language.

### Embedding Norm Distribution

A box‑plot or violin plot showing the distribution of L2 norms across all token embeddings. Tokens with unusually high norms (like `[CLS]` or `[SEP]`) are annotated. This reveals "anisotropy" — the tendency of embeddings to cluster in a narrow cone.

### Nearest‑Neighbor Graph

The WASM similarity module computes the top‑5 nearest neighbors for every token in the vocabulary (or a user‑selected subset). The graph is rendered as a D3 force‑directed graph, where edge thickness = similarity. This is the same concept as the GPT‑2 Token Similarity Visualizer which "visualizes token similarity relationships across GPT‑2 layers ... providing an interactive 'microscope' into how language models process and understand text".

### Semantic Axis Discovery (ICA‑Based)

This is the most novel feature, inspired by the research "Discovering Universal Geometry in Embeddings with ICA" which demonstrates "that each embedding can be expressed as a composition of a few intrinsic interpretable axes and that these semantic axes remain consistent across different languages, algorithms, and modalities".

**Implementation approach:**

1. **WASM ICA module:** An Independent Component Analysis implementation compiled to WebAssembly. ICA differs from PCA in that it finds **statistically independent** components rather than merely uncorrelated ones, which tend to align with interpretable semantic concepts.

2. **Axis discovery workflow:**
   - User selects a subset of the vocabulary (e.g., all nouns, or all emotion‑related words)
   - The ICA module decomposes their embeddings into independent components
   - Each component is tested for correlation with known semantic labels (e.g., sentiment scores from SentiWordNet, concreteness ratings)
   - Components that strongly correlate with known labels are presented as "discovered semantic axes"

3. **Visualization:** Each discovered axis is rendered as a **labeled arrow** in the 3D embedding space. For example, an axis pointing from "sad" tokens to "happy" tokens is labeled "Sentiment (+)". The axes are overlaid on the 3D scatter plot.

4. **User‑defined axes:** Following Parallax's approach which "allows the user to explicitly define the axes of the projection through algebraic formulae", users can also define custom axes by providing word pairs (e.g., "king – man + woman = queen") and the system computes and visualizes the resulting direction.

---

## 🔬 Zone 5: Interpretability — Linear Probe Playground

### Framework: `@wlearn/liblinear` (WebAssembly LIBLINEAR)

The linear probe playground uses LIBLINEAR v2.50 compiled to WebAssembly. This enables **browser‑side training of linear classifiers on embedding vectors** — no server, no GPU, no Python required.

**Workflow:**

1. **Data preparation:** The user selects a classification task from a dropdown (e.g., "POS tagging: Noun vs. Verb", "Sentiment: Positive vs. Negative", or a custom task defined by the user)
2. **Feature extraction:** For the selected tokens, their embedding vectors are extracted from the loaded model's embedding matrix. The WASM module receives these as `Float64Array` typed arrays.
3. **Training:** The `LinearModel` is instantiated with `solver: 'L2R_LR'` (L2‑regularized logistic regression) and fit on the training split. Training completes in milliseconds on a few thousand examples.
4. **Evaluation:** Accuracy, precision, recall, and F1 are displayed. The decision boundary is visualized as a **plane in the 3D embedding space** — tokens correctly classified appear in their true color; misclassified tokens appear in the predicted color with an X marker.
5. **Layer comparison:** Users can train probes on embeddings from different layers of a transformer (layer 0, 3, 6, 9, 12) to see how semantic information becomes more linearly separable in deeper layers.

**Why this works in the browser:** LIBLINEAR uses a dual coordinate descent algorithm that is highly efficient for linear models. When compiled to WebAssembly, it runs at near‑native speed because the hot loop consists of simple arithmetic operations that WASM executes efficiently. The `@wlearn/liblinear` package demonstrates this by providing `model.fit()`, `model.predict()`, `model.predictProba()`, and `model.score()` APIs that all execute entirely client‑side.

### Token Activation Comparison

After fine‑tuning a model (tracked in your `experiment` app), users can compare embeddings before and after fine‑tuning:

1. Load the original model's embeddings and the fine‑tuned model's embeddings
2. For each token, compute the cosine similarity between its original and fine‑tuned embedding
3. Render a **scatter plot** where X = original norm, Y = fine‑tuned norm, color = cosine similarity
4. Tokens that moved most during fine‑tuning (low cosine similarity) are highlighted as "most affected by fine‑tuning"
5. One‑click link to the training run that produced the fine‑tuned model

---

## 💬 Zone 6: Social Features

### "Embedding of the Day"

A community‑curated daily feature where:
1. Users submit interesting embedding relationships they've discovered (e.g., "The embedding of 'dog' is closer to 'cat' in BERT than in GPT‑2")
2. The community votes on submissions
3. The winning submission is featured on the dashboard homepage
4. Submissions include the interactive 3D visualization embedded (not just a screenshot)

### Embedding Visualization Sharing

Every embedding visualization state (camera position, selected tokens, cluster view, projection method) is serialized to a URL parameter. Sharing the URL recreates the exact same view for another researcher. This uses the same pattern as your attention dashboard's sharing system.

### Collections: "Interesting Embedding Clusters"

Users create themed collections of embedding discoveries, similar to GitHub Stars or Pinterest boards. Example collections: "Multilingual Embedding Comparisons" (showing how the same concept clusters in English vs. French vs. Chinese models), "Fine‑tuning Effects" (before/after comparisons), "Semantic Axis Gallery" (showcasing discovered interpretable axes).

---

## 🎮 Zone 7: Gamification — "Embedding Detective"

A daily puzzle game where:

1. The system selects a random token from the vocabulary
2. Its embedding vector is displayed (as a bar chart or radar plot of the top‑20 PCA dimensions)
3. The user must **guess which token** it represents
4. After guessing, the correct token is revealed along with its nearest neighbors
5. **Scoring:** Points awarded based on how close the guess was (measured by cosine similarity between the guessed token's embedding and the target)
6. **Streaks:** Consecutive days played are tracked and displayed
7. **XP rewards:** XP contributes to the user's overall platform level

---

## 🔗 Zone 8: Cross‑Platform Integration Points

The embedding explorer connects to every other part of your platform:

| Platform App | Integration |
|-------------|-------------|
| `experiment` | Load embeddings from any training run; compare before/after fine‑tuning |
| `model_versioning` | Load embeddings from any saved model version |
| `notes` | Save embedding analysis as a researcher note with embedded 3D scene |
| `paper` | Export embedding visualizations as LaTeX figures |
| `community` | Discuss embedding discoveries in linked Q&A threads |
| `portfolios` | Embedding discoveries appear in the researcher's portfolio |
| `leaderboards` | "Embedding Detective" scores on weekly leaderboards |
| `gpu_reputation` | XP from embedding exploration contributes to overall reputation |
| `profiles` | User's embedding expertise badges displayed on profile |
| `messaging` | Share embedding discoveries via private messages |
| `organizations` | Team‑level embedding collections for research groups |

---

## 🏗️ Implementation Roadmap

| Phase | Duration | Components |
|-------|----------|------------|
| **Phase 1: Core Rendering** | Weeks 1‑3 | WebGPU canvas, R3F scene, instanced sphere rendering, orbit controls, camera animation, WASM module loader |
| **Phase 2: Computation** | Weeks 4‑5 | WASM UMAP/PCA/ICA, clustering, similarity search, embedding matrix heatmap |
| **Phase 3: Interactivity** | Weeks 6‑7 | Token search/teleport, trajectory view, cluster hulls, density contours |
| **Phase 4: Analysis** | Weeks 8‑9 | Linear probe playground, semantic axis discovery, embedding statistics dashboard, cosine similarity calculator |
| **Phase 5: Social** | Weeks 10‑11 | Sharing, collections, "Embedding of the Day", gamification |
| **Phase 6: Integration** | Weeks 12‑13 | Cross‑platform links (experiments, notes, paper, community) |

---

## 📊 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **3D Rendering** | Three.js via R3F + WebGPU | GPU‑accelerated sphere cloud, camera, effects |
| **2D Rendering** | HTML5 Canvas + OffscreenCanvas | Embedding matrix heatmap at scale |
| **WASM Compute** | Rust → wasm‑pack | UMAP, PCA, ICA, clustering, similarity |
| **Linear Classification** | `@wlearn/liblinear` (WASM) | Browser‑side linear probe training |
| **Data Analysis** | DuckDB‑WASM | Client‑side SQL queries on embedding metadata |
| **State Management** | Zustand + immer | Reactive embedding state, undo/redo |
| **Animation** | `framer-motion` + GSAP | Math panel step animations, camera transitions |
| **Charts** | `recharts` + `d3` | Statistics dashboard charts |
| **Math Rendering** | `katex` / `react-latex` | Live LaTeX formula display |
| **Backend** | Django REST API | Model loading, embedding extraction, experiment data |
| **Real‑time** | WebSocket / Channels | Collaborative exploration, live updates |

The entire system operates with **zero data upload** — all computation happens client‑side through WebAssembly, following Apple's "privacy‑preserving, reproducible" design that "completes all computational tasks including embedding generation and projection locally within the browser". This architecture makes the Embedding Explorer both fast and privacy‑respecting, suitable for researchers working with proprietary models or sensitive text data.