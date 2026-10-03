# 🥽 VR Immersion Room — "Step Inside the Transformer"

## Complete Implementation Specification

---

## TABLE OF CONTENTS

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack & Framework Selection](#2-technology-stack--framework-selection)
3. [Core Rendering Pipeline](#3-core-rendering-pipeline)
4. [Scene Graph & Spatial Organization](#4-scene-graph--spatial-organization)
5. [Token Representation & Visualization](#5-token-representation--visualization)
6. [Attention Connection Rendering](#6-attention-connection-rendering)
7. [Layer Transition System (Floors)](#7-layer-transition-system-floors)
8. [VR Input & Interaction System](#8-vr-input--interaction-system)
9. [GPU Compute Particle System](#9-gpu-compute-particle-system)
10. [Multi-User Collaboration & Networking](#10-multi-user-collaboration--networking)
11. [Spatial Audio System](#11-spatial-audio-system)
12. [UI Panels in VR (Diegetic Interfaces)](#12-ui-panels-in-vr-diegetic-interfaces)
13. [Performance Optimization Strategy](#13-performance-optimization-strategy)
14. [Device Compatibility Matrix](#14-device-compatibility-matrix)
15. [Session Recording & Playback](#15-session-recording--playback)
16. [Implementation Roadmap](#16-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 System Architecture Diagram

The VR Immersion Room follows a layered architecture with clear separation between the VR rendering engine, the data pipeline, the networking layer, and the platform integration points:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VR IMMERSION ROOM ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────── CLIENT LAYER ───────────────────────────┐ │
│  │                                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────┐    │ │
│  │  │              React Three Fiber <Canvas>                  │    │ │
│  │  │  ┌──────────────────────────────────────────────────┐   │    │ │
│  │  │  │           @react-three/xr <XR> Provider           │   │    │ │
│  │  │  │  ┌─────────────┐  ┌──────────┐  ┌─────────────┐  │   │    │ │
│  │  │  │  │ Immersion   │  │ Hand     │  │ Controller  │  │   │    │ │
│  │  │  │  │ Room Scene  │  │ Tracking │  │ Input       │  │   │    │ │
│  │  │  │  └─────────────┘  └──────────┘  └─────────────┘  │   │    │ │
│  │  │  └──────────────────────────────────────────────────┘   │    │ │
│  │  └─────────────────────────────────────────────────────────┘    │ │
│  │                                                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │                  Web Worker Pool                          │   │ │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │   │ │
│  │  │  │ Layout       │ │ Embedding    │ │ Attention Matrix │  │   │ │
│  │  │  │ Compute      │ │ Projection   │ │ Stream Decoder   │  │   │ │
│  │  │  │ (t-SNE/UMAP) │ │ (PCA/GPU)    │ │ (WebSocket)      │  │   │ │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────┘  │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  │                                                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │              WebGPU Compute Pipeline                       │   │ │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │   │ │
│  │  │  │ Particle     │ │ Connection   │ │ Attention Flow   │  │   │ │
│  │  │  │ Physics      │ │ Geometry     │ │ Field Sim        │  │   │ │
│  │  │  │ (Compute)    │ │ (Vertex)     │ │ (Compute)        │  │   │ │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────┘  │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────── NETWORK LAYER ──────────────────────────┐ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │ WebSocket    │  │ WebRTC       │  │ Yjs CRDT             │  │ │
│  │  │ (State Sync) │  │ (P2P Avatars)│  │ (Shared State)       │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────── DJANGO INTEGRATION ─────────────────────┐ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │ Attention    │  │ Session      │  │ Collaboration         │  │ │
│  │  │ Data API     │  │ Recording    │  │ Room Management       │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

The data flows through five distinct stages:

1. **Fetch Stage**: Attention data is loaded from the Django REST API (`/api/experiments/attention/`) or streamed via WebSocket for live training visualization. The data includes tokens, attention matrices per layer/head, hidden states for embedding projection, and model metadata.

2. **Process Stage**: Web Workers compute t-SNE/UMAP projections for 3D token positioning, PCA for dimensionality reduction of Q/K/V vectors, and layout algorithms for the spatial arrangement of layers. This prevents blocking the main thread.

3. **Render Stage**: React Three Fiber constructs the scene graph. The `<XR>` provider from `@react-three/xr` wraps the entire scene for VR support. GPU compute shaders handle particle physics for attention flow visualization.

4. **Interact Stage**: Hand tracking data from WebXR (via `@react-three/xr`'s `<Hands>` component) enables grabbing, pointing, and gesturing. Eye tracking (where available via HeedVision's approach) records where users look.

5. **Sync Stage**: For multi-user sessions, Yjs CRDT synchronizes shared state (which tokens are selected, which layer the group is viewing). WebRTC DataChannels stream avatar positions and hand poses peer-to-peer with minimal latency.

---

## 2. Technology Stack & Framework Selection

### 2.1 Core Rendering Engine

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **3D Engine** | Three.js | r168+ | Industry-standard WebGL/WebGPU renderer with the largest ecosystem and best WebXR support |
| **React Integration** | React Three Fiber (R3F) | v9+ | Declarative, component-based 3D scene construction; "Craft immersive XR educational experiences through declarative 3D rendering" |
| **XR Provider** | @react-three/xr | v6+ | WebXR manager that "configures your scene for XR rendering and interaction" |
| **Hand Tracking** | @react-three/xr `<Hands>` | — | "Add hands model for hand-tracking. Works out of the box on Oculus Browser v13" |
| **Helper Library** | @react-three/drei | v9+ | Provides `<PositionalAudio>`, `<Text>`, `<Environment>`, `<Gltf>` and many other utilities |

### 2.2 GPU Compute & Particle System

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **GPU API** | WebGPU | Latest | "WebGPU通过计算着色器（Compute Shader）直接调用GPU的并行计算能力" — 3-5x faster than WebGL for particle systems |
| **Compute Shaders** | WGSL (WebGPU Shading Language) | — | Type-safe shader language; "支持自动导数计算，比WebGL快3-5倍（实测10万粒子系统可达300FPS）" |
| **TSL** | Three.js Shading Language | — | "GPU-accelerated galaxy simulation using WebGPU, Three.js, and TSL" |
| **Buffer Strategy** | GPUBuffer (Storage Buffer) | — | "stores the particles in a GPUBuffer instead of a texture" for efficient compute-to-render pipeline |

### 2.3 Multi-User & Networking

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **State Sync** | Yjs (CRDT) | "using Three.js for 3D graphics, Yjs for data synchronization, and WebSockets for real-time communication" |
| **Avatar Sync** | WebRTC DataChannel (PeerJS) | "Players connect directly through P2P with no server in between" |
| **Room Management** | Django Channels | Existing infrastructure — WebSocket groups for room membership |
| **Server Relay** | Node.js WebSocket fallback | For users behind symmetric NAT where P2P fails |

### 2.4 Spatial Audio

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Audio Engine** | Web Audio API + Three.js `PositionalAudio` | "精确的空间音效定位 - 动态音量衰减控制" |
| **HRTF** | Web Audio API `PannerNode` | Built-in head-related transfer function for spatial perception |
| **Helper** | @react-three/drei `<PositionalAudio>` | "使用`@react-three/drei`中的`PositionalAudio`组件，为游戏添加空间音频效果" |

### 2.5 Data Processing (Web Workers)

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **t-SNE** | `@aspect-dev/tsne` or custom WASM | Dimensionality reduction for token positioning |
| **UMAP** | `umap-js` | Faster alternative to t-SNE for large token sets |
| **Matrix Ops** | `gpu.js` or WebGPU compute | Parallel matrix operations for attention weight processing |

### 2.6 Platform Integration (Django Backend)

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Attention Data API** | Django REST Framework | Existing `/api/experiments/attention/` endpoint |
| **Live Streaming** | Django Channels WebSocket | Existing WebSocket infrastructure for real-time data |
| **Session Persistence** | Django ORM | Save VR session recordings, annotations, and collaboration logs |
| **Authentication** | DRF Token Auth | Same auth as the rest of the platform |

---

## 3. Core Rendering Pipeline

### 3.1 Canvas Setup

The root `<Canvas>` component from React Three Fiber must be configured with WebGPU support and VR-ready settings:

**Key configuration parameters:**
- `gl` prop set to use WebGPU backend when available, falling back to WebGL2
- `frameloop="demand"` to only render when needed (saves battery on mobile/standalone headsets)
- `dpr={[1, 1.5]}` for Quest 3 — lower pixel ratio for performance; `dpr={[1, 2]}` for desktop VR
- `shadows="soft"` disabled in VR (too expensive)
- `camera={{ position: [0, 1.6, 0], fov: 90 }}` — human height, wide FOV for VR

### 3.2 XR Provider Setup

The `<XR>` component from `@react-three/xr` wraps the entire scene:

**XR features to enable:**
- `referenceSpace="local-floor"` — user stands on the floor, can walk around
- `handTracking` — enables hand skeletal tracking
- `controllerTracking` — enables motion controllers as fallback
- `eyeTracking` (experimental) — where supported by headset (Apple Vision Pro, Quest Pro)
- `foveatedRendering` — renders center of view at full resolution, periphery at lower resolution (Quest 2/3/Pro)

**Non-VR fallback mode:** When no headset is detected, the scene renders in "explore mode" with mouse/keyboard controls (OrbitControls for rotation, WASD for movement, scroll for zoom). This is critical — most users will first experience the immersion room on desktop before putting on a headset.

### 3.3 Rendering Order & Layer Composition

The scene uses Three.js render layers to compose the final image:

1. **Layer 0 — Environment**: Skybox/sphere, ambient lighting, floor grid
2. **Layer 1 — Token Nodes**: Spheres and text labels for each token
3. **Layer 2 — Attention Connections**: Curved tubes, particle flows, laser beams
4. **Layer 3 — Layer Floors**: Transparent rings representing each transformer layer
5. **Layer 4 — UI Overlay**: Diegetic panels, tooltips, selection highlights
6. **Layer 5 — Avatar Representation**: Other users' avatars and gaze indicators

For VR headsets that support WebXR Layers API (composition layers), static UI panels can be rendered at native resolution on a separate quad layer: "a static cylinder or quad layer can be drawn once at high resolution and is resampled only once by the compositor".

---

## 4. Scene Graph & Spatial Organization

### 4.1 Overall Scene Structure

```
ImmersionRoom (THREE.Group)
├── EnvironmentGroup
│   ├── SkySphere (panoramic gradient)
│   ├── AmbientLight
│   ├── PointLights (3x, warm)
│   └── FloorGrid (semi-transparent, gives spatial reference)
├── LayerFloorGroup[]
│   ├── RingMesh (transparent ring per layer)
│   ├── LayerLabel (Text)
│   └── TokenNodeGroup[]
│       ├── TokenSphere
│       ├── TokenLabel
│       ├── EmbeddingVectorIndicator (small arrow)
│       └── SelectionGlow (on hover/select)
├── ConnectionGroup
│   ├── TubeConnection[]
│   ├── ParticleFlowConnection[]
│   └── LaserBeamConnection[] (for strongest attention)
├── UIOverlayGroup
│   ├── ControlPanel (diegetic)
│   ├── TooltipLabel
│   └── MiniMapIndicator
└── AvatarGroup (multi-user)
    ├── HeadMesh[]
    ├── HandMesh[]
    └── GazeBeam[]
```

### 4.2 Spatial Layout Algorithm

The spatial layout is computed by a Web Worker to avoid blocking the main thread:

**Token positioning** uses t-SNE or UMAP projection of the hidden states (or, for a simpler initial approach, positions tokens on a circular ring within each layer). The ring radius is proportional to the number of tokens — larger for longer sequences.

**Layer spacing** is uniform (e.g., 3 meters between layers). The user starts at Layer 0 and can "ascend" or "descend" through the model. The current layer is highlighted with a brighter ring.

**Connection routing** uses quadratic Bezier curves: the control point is offset perpendicular to the ring, creating arcs that rise above/below the layer plane. Thicker connections (higher attention) arc higher, making them visually prominent.

### 4.3 Reference Space Configuration

The WebXR reference space is `"local-floor"` — the user's physical floor is the origin. The virtual scene is positioned so that the user stands in the center of the token ring at the first layer. Teleportation targets are placed at each layer for quick navigation.

---

## 5. Token Representation & Visualization

### 5.1 Token Node Design

Each token is represented as a **sphere** with a **text label** floating above it:

- **Sphere radius**: Proportional to the token's "importance" (e.g., max attention received across all heads, or its norm in the hidden state)
- **Sphere material**: `MeshStandardMaterial` with `roughness=0.3`, `metalness=0.1`
- **Sphere color**: Mapped by token position (HSL gradient along the sequence) or by part-of-speech (noun=blue, verb=red, etc.) using the Dodrio-style linguistic annotation overlay
- **Text label**: Rendered using `@react-three/drei`'s `<Text>` component with `fontSize` scaled by distance. The text shows the token string (truncated to 8 characters for long tokens). The text always faces the camera (billboarding) for readability.

### 5.2 Token Interaction States

| State | Visual Effect | Trigger |
|-------|--------------|---------|
| **Idle** | Subtle glow, normal scale | Default |
| **Hovered** | Brighten, scale 1.2x, show tooltip | Gaze or controller ray hover |
| **Selected** | Pulsing ring, emit connections highlight | Click / pinch gesture |
| **Grabbed** | Follow hand position, semi-transparent | Grab gesture (pinch + hold) |
| **Attention Source** | Emitting particles | Being queried (row of attention matrix) |
| **Attention Target** | Receiving particles, brightening | Being keyed (column of attention matrix) |

### 5.3 Embedding Vector Visualization

When a token is "grabbed" (pinched), a small 3D arrow appears showing its embedding vector direction. For tokens in the same layer, the cosine similarity between their embeddings is visualized as the angle between their arrows — users can physically align tokens to compare their embeddings. This is a direct spatial analog of the dot product in attention computation.

---

## 6. Attention Connection Rendering

### 6.1 Three Rendering Modes

The attention connections between tokens can be rendered in three modes, user-switchable via a diegetic panel:

**Mode A: Tube Geometry (Default)**
- Uses Three.js `TubeGeometry` with a `QuadraticBezierCurve3`
- Tube radius proportional to attention weight
- Material: `MeshBasicMaterial` with vertex colors (gradient from query color to key color)
- Performance: ~5,000 connections at 60fps on Quest 3

**Mode B: Particle Flow (GPU Compute)**
- Uses WebGPU compute shader to simulate particles flowing along Bezier paths
- Each attention connection spawns `N` particles where `N ∝ weight`
- Particles are rendered as point sprites with additive blending
- Performance: ~100,000 particles at 60fps using WebGPU compute + draw indirect
- Visual effect: "attention as flowing energy" — particles stream from query tokens to key tokens, creating a dynamic, alive visualization

**Mode C: Laser Beam (Highest Attention Only)**
- For the top-10 strongest attention connections in the current view
- Uses a `CylinderGeometry` (thin tube) with emissive material and bloom post-processing
- Creates dramatic "laser" lines that are visually striking in VR
- Performance: negligible (only 10 connections)

### 6.2 Particle Flow Physics (WebGPU Compute Shader)

The particle flow is the most visually impressive mode and deserves detailed specification:

**Compute Shader Structure (WGSL):**

The shader runs one workgroup per attention connection, with each workgroup processing 64 particles:

1. **Input Buffers**: Particle positions (vec3), velocities (vec3), lifetimes (f32), Bezier control points per connection (vec3 × 4)
2. **Per-Frame Update**: Each particle advances along its Bezier curve based on velocity. When a particle reaches the end of the curve (reaches the key token), it respawns at the start (query token) with a small random offset.
3. **Velocity Variation**: Particles at the center of the beam (higher attention weight) move faster; edge particles move slower, creating a natural "core flow" appearance.
4. **Color Computation**: Each particle's color is interpolated between the query token color and key token color based on its position along the curve.

**Rendering Pipeline:**
1. Compute shader dispatches particle updates (GPU → GPU, no CPU readback)
2. Vertex shader reads particle positions from the storage buffer
3. Fragment shader renders each particle as a soft circle (distance-based alpha) with additive blending
4. A post-processing bloom pass (using `@react-three/postprocessing`) makes the particles glow

**Performance target**: 100,000 particles at 300 FPS on desktop GPU (WebGPU), or 50,000 particles at 72 FPS on Quest 3 (WebGL2 compute fallback).

### 6.3 Connection Culling & LOD

For scenes with many tokens (e.g., 128+ token sequence), not all connections can be rendered. A level-of-detail system applies:

- **Distance-based culling**: Connections further than a threshold distance from the user are not rendered
- **Weight-based filtering**: Below a minimum attention weight (configurable via slider), connections are hidden
- **Head-based filtering**: User selects which heads to visualize; only connections from selected heads appear
- **Token-pair filtering**: When a user selects a specific query token, only connections from that token are shown (all others dim to 10% opacity)

---

## 7. Layer Transition System (Floors)

### 7.1 Layer Ring Design

Each transformer layer is represented as a **transparent ring** (torus) at a specific Y-height in the scene. The ring serves as both a spatial reference and a navigation target:

- **Ring radius**: Slightly larger than the token ring (e.g., 4.5m vs 4.0m for tokens)
- **Ring material**: Semi-transparent (`opacity: 0.3`) with emissive glow
- **Ring color**: Layer index mapped to HSL gradient (Layer 0 = blue, Layer N = red)
- **Layer label**: Large floating text above each ring showing "Layer N"

### 7.2 Ascending/Descending Through Layers

The user navigates between layers using:

**In VR:**
- **Teleport beam**: Point controller at a layer ring and press trigger to teleport
- **Thumbstick vertical**: Push thumbstick up/down to smoothly "elevator" between layers
- **Hand gesture**: Palm-up → ascend one layer; palm-down → descend one layer

**In desktop explore mode:**
- **Scroll wheel**: Ascend/descend layers
- **Page Up/Page Down keys**: Jump between layers
- **Layer selector panel**: Click on a layer in the minimap

### 7.3 Cross-Layer Attention Visualization

When the user stands at Layer N and looks at tokens in Layer N+1, cross-layer attention connections (residual stream flow) appear as **vertical arcs** connecting tokens across layers. This visualizes the "residual stream" concept — the information flowing through the model. The arcs use a different color (gold/amber) to distinguish them from intra-layer attention connections.

---

## 8. VR Input & Interaction System

### 8.1 Hand Tracking Implementation

Hand tracking is implemented via `@react-three/xr`'s `<Hands>` component, which uses the WebXR Hand Tracking API: "Controllers can be added with `<Controllers />` for motion-controllers and/or `<Hands />` for hand-tracking".

**Joint data structure:** The OpenXR hand tracking API provides 25 joint positions per hand (wrist, 4 joints per finger including metacarpal, proximal, intermediate, distal, and tip). The `XR_EXT_hand_tracking_data_source` extension (available in Monado 25.1.0+) "enables distinguishing between unobstructed and conforming hand-tracking modes" for more robust tracking.

**Hand mesh rendering:** Each joint is rendered as a small sphere. Joints are connected by cylinder segments to form a skeletal hand. The hand mesh uses a semi-transparent material with a subtle glow. When the hand is near an interactive object, the fingertip spheres brighten to provide visual feedback.

### 8.2 Gesture Recognition System

Gestures are recognized using a finite state machine that processes joint positions each frame:

| Gesture | Joint Configuration | Action |
|---------|-------------------|--------|
| **Point** | Index finger extended, others curled | Raycast to select/highlight tokens |
| **Pinch (Grab)** | Thumb tip + index tip distance < 2cm | Grab selected token, move it in space |
| **Pinch + Pull** | Grab + move hand apart | "Stretch" an attention connection to see its weight |
| **Palm Up** | Palm normal facing upward | Ascend one layer |
| **Palm Down** | Palm normal facing downward | Descend one layer |
| **Fist** | All fingers curled | Release grabbed object / cancel selection |
| **Pinch + Rotate** | Grab + twist wrist | Rotate the token ring (change viewing angle) |
| **Two-Hand Spread** | Both hands, move apart | Zoom in (scale the scene) |
| **Two-Hand Pinch** | Both hands, move together | Zoom out |

### 8.3 Raycasting for Selection

The pointing gesture casts a ray from the index fingertip. The ray is visualized as a thin laser line. Objects within a small radius of the ray are highlighted. When the user pinches (thumb + index), the closest highlighted object is "grabbed" — it follows the hand position with physics-based smoothing.

### 8.4 Controller Fallback

When hand tracking is unavailable (or for users who prefer controllers), the standard VR controller with trigger/grip/thumbstick is supported. The controller model is rendered with a laser pointer emanating from the front. All gestures have controller equivalents (e.g., trigger = pinch, grip = grab).

### 8.5 Desktop Mouse/Keyboard Fallback

For non-VR users, standard FPS controls apply:
- **Mouse**: Look around (pointer lock)
- **WASD**: Move through the scene
- **Left click**: Select token
- **Right click + drag**: Rotate view
- **Scroll**: Zoom in/out or ascend/descend layers
- **Shift + click**: "Grab" and move a token

---

## 9. GPU Compute Particle System

### 9.1 Why GPU Compute?

The attention visualization requires rendering thousands of flowing particles representing attention weights. On CPU, updating 100,000 particles per frame would consume ~15ms (below 60fps). WebGPU compute shaders can process all particles in parallel on the GPU in <1ms: "WebGPU通过计算着色器（Compute Shader）直接调用GPU的并行计算能力，显著提升了科学数据的处理效率。例如，在流体动力学模拟中，WebGPU可将粒子系统的更新计算从CPU卸载到GPU，实现毫秒级渲染迭代".

### 9.2 Buffer Architecture

The particle system uses a triple-buffer approach to avoid synchronization stalls:

1. **Particle State Buffer (Storage Buffer)**: Contains position (vec3f32), velocity (vec3f32), lifetime (f32), connectionID (u32), seed (u32) for each particle. Size = `numParticles × 32 bytes`.
2. **Connection Metadata Buffer (Uniform Buffer)**: Contains the Bezier control points (vec3f32 × 4), query token color (vec3f32), key token color (vec3f32), and weight (f32) for each attention connection. Size = `numConnections × 64 bytes`.
3. **Indirect Draw Buffer**: Contains `vertexCount`, `instanceCount`, `firstVertex`, `firstInstance` — written by the compute shader for GPU-driven rendering.

### 9.3 Compute Shader Dispatch

The compute shader is dispatched with `ceil(numParticles / 256)` workgroups of 256 threads each. Each thread:
1. Reads its particle state from the Storage Buffer
2. Reads the connection metadata for its assigned connection
3. Updates position by advancing along the Bezier curve: `P(t + Δt) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃`
4. Checks if the particle has reached the end (t ≥ 1.0) — if so, resets to t = 0 with random jitter
5. Writes updated state back to the Storage Buffer

### 9.4 Rendering from Compute Output

The vertex shader reads particle positions directly from the Storage Buffer (no CPU readback). Each particle is rendered as a camera-facing quad (billboard) using `THREE.Points` or a custom `InstancedMesh`. The fragment shader applies a soft circular alpha mask and blends additively. A post-processing bloom pass (via `@react-three/postprocessing`) creates the glow effect.

### 9.5 WebGPU Feature Detection & Fallback

The system detects WebGPU support via `navigator.gpu`. If available, compute shaders are used. If not (e.g., on older devices or browsers without WebGPU), fallback strategies apply:
- **WebGL2 Transform Feedback**: Emulates compute by using vertex shader to update positions
- **CPU fallback with WASM**: Rust compiled to WebAssembly for particle physics, running in a Web Worker
- **Reduced particle count**: Automatically reduces `maxParticles` to maintain 60fps

---

## 10. Multi-User Collaboration & Networking

### 10.1 Architecture

The multi-user system enables multiple researchers to explore the same attention visualization together in VR. The architecture follows a hybrid client-server + peer-to-peer model:

- **Django Channels (Server)**: Room management, initial state sync, session recording
- **WebRTC DataChannel (P2P)**: Avatar pose streaming (head position, hand joints, gaze direction), low-latency interaction sync
- **Yjs CRDT (Shared State)**: Conflict-free replicated data type for shared selection state, annotation pins, and layer position. "Using Three.js for 3D graphics, Yjs for data synchronization, and WebSockets for real-time communication"

### 10.2 Avatar Representation

Each remote user is represented as a simplified avatar:
- **Head**: A small sphere at the user's headset position, colored uniquely per user
- **Hands**: Skeletal hand meshes that mirror the user's real hand poses
- **Gaze beam**: A thin laser from the head position showing where the user is looking (inspired by HeedVision: "CAAVs that track, record, and revisualize the collective attention of multiple users over time")
- **Name label**: Floating text above the head showing the user's name

### 10.3 Collaborative Interaction

Shared interactions include:

| Interaction | Sync Method | Latency Target |
|-------------|-------------|----------------|
| Token selection | Yjs shared state | <100ms |
| Layer navigation | Yjs shared state | <100ms |
| Annotation pins | Yjs shared state (persistent) | <500ms |
| Avatar pose | WebRTC DataChannel | <50ms |
| Voice chat | WebRTC audio track | <200ms |
| Hand gestures | WebRTC DataChannel | <50ms |

### 10.4 Session Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Explore Together** | All users freely navigate; shared selection state | Casual exploration |
| **Follow Mode** | One user is the "guide"; others automatically follow their viewpoint | Teaching, presentations |
| **Presentation Mode** | Guide controls what everyone sees; audience can raise hands to ask questions | Conference demos |
| **Annotation Mode** | Users place persistent annotation pins; threaded discussion per pin | Research collaboration |

---

## 11. Spatial Audio System

### 11.1 Audio Sources

The spatial audio system uses Three.js `PositionalAudio` objects attached to 3D positions in the scene. The Web Audio API's `PannerNode` provides head-related transfer function (HRTF) processing for spatial perception.

**Audio sources in the scene:**

| Source | Position | Sound | Purpose |
|--------|----------|-------|---------|
| **Token nodes** | Each token's 3D position | Soft chime when attention is high | Auditory indication of important tokens |
| **Layer transitions** | Layer ring positions | Whoosh sound when ascending/descending | Spatial feedback for navigation |
| **Connection flow** | Midpoint of Bezier curves | Variable-pitch hum (pitch ∝ attention weight) | Auditory attention "weight map" |
| **Ambient** | Scene center | Low drone, subtle | Background atmosphere |

### 11.2 Attention Sonification

Each attention connection produces a subtle sound whose characteristics encode the attention weight:

- **Pitch**: Higher attention → higher pitch. Mapped logarithmically: `pitch = 220Hz × (1 + weight × 4)` (range: 220Hz–1100Hz)
- **Volume**: Higher attention → louder. Mapped: `volume = -40dB + weight × 30dB`
- **Timbre**: Positional heads → pure sine tone; semantic heads → rich harmonic series
- **Panning**: Sound source positioned at the midpoint of the attention arc, so the user hears it spatially

### 11.3 Voice Chat (Multi-User)

WebRTC audio tracks are captured from the user's microphone (with permission). Each remote user's audio is spatialized at their avatar's head position: "I added 3D positional audio using Three.js's PositionalAudio. Depending on your seat's location, the sound gets softer, louder, or shifts left/right". This means you hear other researchers' voices coming from where they're standing in the virtual room.

---

## 12. UI Panels in VR (Diegetic Interfaces)

### 12.1 Diegetic vs. Non-Diegetic UI

The VR immersion room uses **diegetic interfaces** — UI elements that exist within the 3D world rather than being overlaid on the screen. This is more immersive and natural in VR.

**Diegetic panels** (floating in 3D space):
- Control panel (attached to the user's off-hand, like a wristwatch)
- Token information panel (appears next to selected/gazed token)
- Layer selector (floating ring near the user)
- Minimap (small 3D overview of all layers, like a tabletop model)

**Non-diegetic** (only in desktop fallback mode):
- Settings menu overlay
- Performance stats (FPS, particle count)
- Session recording controls

### 12.2 Wrist-Mounted Control Panel

The user's non-dominant hand has a small panel "attached" to it (like a smartwatch). It shows:
- Current layer number / total layers
- Active head selection
- Connection mode toggle (Tubes / Particles / Laser)
- Particle density slider
- Session recording button

The panel uses a `PlaneGeometry` rendered to a `MeshStandardMaterial` with a canvas texture (using Three.js `CanvasTexture`). The canvas is drawn using standard HTML Canvas 2D API, then updated as a texture each frame. This approach is compatible with React Three Fiber via `useFrame`.

### 12.3 Token Information Popup

When a user gazes at (or points at) a token, a small information panel appears floating next to the token. It shows:
- Token text
- Token position in sequence
- Top-3 attention targets (which tokens does this token attend to most?)
- Top-3 attention sources (which tokens attend to this token most?)
- Hidden state norm
- If available: Dodrio-style linguistic annotations (POS tag, dependency role)

### 12.4 Minimap (Tabletop Model)

A small 3D model of the entire transformer sits on a "table" at the center of the room (or floats at waist height). It shows all layers stacked vertically, with tokens as tiny dots. The user's current position and gaze direction are highlighted. Other users' positions are also shown. The minimap can be "grabbed" and resized, rotated, or moved.

---

## 13. Performance Optimization Strategy

### 13.1 Rendering Performance Targets

| Platform | Target FPS | Max Particles | Max Connections | Max Tokens |
|----------|-----------|---------------|-----------------|------------|
| Desktop (RTX 4090, WebGPU) | 120 | 500,000 | 10,000 | 512 |
| Desktop (GTX 1660, WebGL2) | 60 | 50,000 | 2,000 | 256 |
| Quest 3 (Standalone) | 72 | 20,000 | 500 | 128 |
| Quest 2 (Standalone) | 72 | 10,000 | 200 | 64 |
| Apple Vision Pro (WebXR) | 90 | 50,000 | 1,000 | 256 |
| Mobile (AR mode) | 30 | 5,000 | 100 | 32 |

### 13.2 Optimization Techniques

**Frustum Culling**: Tokens and connections outside the user's field of view are not rendered. Three.js does this automatically for mesh objects.

**Occlusion Culling**: Tokens hidden behind other tokens or layer rings are not rendered. Implemented via a simple distance-based check (if token A is within 0.5m of token B and behind it, skip rendering A).

**LOD (Level of Detail)**:
- **High detail** (within 2m of user): Full text labels, sphere with 32 segments, emissive glow, particle flows
- **Medium detail** (2m–10m): Smaller text labels, sphere with 16 segments, simplified connections
- **Low detail** (10m+): Dots only, no labels, aggregate connections shown as a "cloud"

**GPU Instancing**: Tokens are rendered using `THREE.InstancedMesh` — a single draw call for all token spheres. Colors and transforms are stored in instance buffers updated each frame.

**Texture Atlas for Labels**: All token text labels are rendered to a single texture atlas using Canvas 2D. Each label is a quad mapped to its region of the atlas, reducing draw calls from N to 1.

**Foveated Rendering** (VR headsets that support it): The center 20° of the user's view is rendered at full resolution; the periphery at 1/4 resolution. This can yield 30-50% GPU savings on Quest Pro and Apple Vision Pro.

**Web Worker Offloading**: t-SNE/UMAP computation, matrix operations, and layout algorithms run in Web Workers, never on the main thread.

### 13.3 Memory Budget

| Buffer | Size (for 100K particles, 1K connections) |
|--------|-------------------------------------------|
| Particle State Buffer | 100,000 × 32B = 3.2 MB |
| Connection Metadata Buffer | 1,000 × 64B = 64 KB |
| Token Position Buffer | 256 × 12B = 3 KB |
| Instance Matrix Buffer | 256 × 64B = 16 KB |
| Label Texture Atlas | 2048×2048 RGBA = 16 MB |
| **Total GPU Memory** | **~20 MB** |

This is well within the memory budget of even standalone VR headsets (Quest 3 has 8GB shared RAM).

---

## 14. Device Compatibility Matrix

### 14.1 VR Headsets

| Headset | WebXR Support | Hand Tracking | Eye Tracking | Performance Tier |
|---------|--------------|---------------|--------------|------------------|
| **Meta Quest 3** | Full (Oculus Browser) | Yes (v13+) | No | Medium |
| **Meta Quest 3S** | Full | Yes | No | Medium-Low |
| **Meta Quest Pro** | Full | Yes | Yes | Medium |
| **Apple Vision Pro** | Full (Safari) | Yes (excellent) | Yes | High |
| **HTC Vive XR Elite** | Full (Wolvic) | Yes | No | Medium |
| **Pico 4** | Full | Yes | No | Medium |
| **Valve Index** | Via SteamVR WebXR | No (controllers only) | No | High (PC-tethered) |
| **Magic Leap 2** | Partial | Yes | Yes | Medium-Low |

### 14.2 Desktop Browsers

| Browser | WebGPU Support | WebXR Support | Notes |
|---------|---------------|---------------|-------|
| **Chrome 125+** | Full | Emulated | WebGPU + WebGL2 fallback |
| **Edge 125+** | Full | Emulated | Same as Chrome |
| **Firefox 130+** | Experimental flag | Emulated | May need `dom.webgpu.enabled` |
| **Safari 18+** | Experimental flag | Emulated (Vision Pro native) | Best on Apple Silicon |

### 14.3 Feature Detection & Graceful Degradation

The system detects capabilities at startup and adjusts accordingly:

```pseudocode
if (navigator.gpu) → Use WebGPU compute shaders for particles
else if (WEBGL2_compute) → Use WebGL2 transform feedback
else → Use CPU + WASM for particles, reduced count

if (navigator.xr) → Enable full VR mode with hand tracking
else → Desktop explore mode with mouse/keyboard

if (handTracking supported) → Enable gesture recognition
else → Use controllers or mouse/keyboard

if (eyeTracking supported) → Enable foveated rendering + gaze beams
```

---

## 15. Session Recording & Playback

### 15.1 Recording Data

Each VR session can be recorded for later playback, sharing, or teaching. The recording captures:

- **Headset pose** (position, orientation) at 72Hz
- **Hand joint poses** (25 joints × 2 hands) at 72Hz
- **Voice audio** (if enabled) as a WebM/Opus stream
- **Interaction events**: token selections, layer changes, annotation placements, gesture events
- **Scene state**: which model, layer, head, and text are being visualized

The recording is compressed (poses use delta encoding, audio uses Opus codec) and uploaded to the Django backend for storage. A 10-minute session with 2 users produces approximately 15-30 MB of data.

### 15.2 Playback

Playback runs entirely in the browser (desktop or VR). The recorded poses are replayed through the same React Three Fiber scene. The viewer can:
- Watch from the recorder's perspective (first-person)
- Watch from any angle (free camera)
- Watch from a "god's eye" overview
- Pause, rewind, fast-forward
- Click on tokens during playback to see what the recorder was looking at

### 15.3 Integration with Platform

Recorded sessions are stored as `VRRecording` models in Django, linked to the user's profile and the attention pattern being explored. Sessions can be:
- Shared via link (like the existing share-link system)
- Embedded in notes or papers (using the existing widget embed)
- Used as teaching materials in courses
- Reviewed collaboratively (multiple people watching the same recording with synced playback)

---

## 16. Implementation Roadmap

### Phase 1: Core Rendering (Weeks 1-3)

**Goal**: A basic 3D scene with tokens and attention connections, viewable on desktop.

| Task | Duration | Dependencies |
|------|----------|-------------|
| Set up React Three Fiber canvas with basic scene | 2 days | None |
| Implement token node rendering (spheres + labels) | 3 days | R3F canvas |
| Implement Bezier curve connection rendering | 3 days | Token rendering |
| Implement layer ring system | 2 days | Token rendering |
| Add orbit controls for desktop navigation | 1 day | R3F canvas |
| Integrate with attention data API | 2 days | Django backend |
| Add basic layer/head selector UI | 2 days | Data integration |

### Phase 2: GPU Particles & XR (Weeks 4-6)

**Goal**: Particle flow rendering and basic VR support.

| Task | Duration | Dependencies |
|------|----------|-------------|
| Set up WebGPU compute shader pipeline | 3 days | Phase 1 |
| Implement particle physics compute shader | 3 days | WebGPU setup |
| Implement particle rendering (point sprites) | 2 days | Compute shader |
| Add bloom post-processing | 1 day | Particle rendering |
| Set up @react-three/xr provider | 2 days | Phase 1 |
| Implement VR hand tracking display | 2 days | XR provider |
| Implement teleport navigation | 1 day | XR provider |
| Add desktop fallback mode | 1 day | XR provider |

### Phase 3: Interaction & Polish (Weeks 7-9)

**Goal**: Full VR interaction, gesture recognition, and performance optimization.

| Task | Duration | Dependencies |
|------|----------|-------------|
| Implement gesture recognition FSM | 3 days | Hand tracking |
| Implement token grab and move | 2 days | Gesture recognition |
| Implement connection inspection (grab + pull) | 2 days | Gesture recognition |
| Add diegetic UI panels (wrist panel, token popup) | 3 days | Phase 2 |
| Implement LOD system | 2 days | Phase 2 |
| Performance profiling and optimization | 3 days | All features |
| Quest 3 testing and tuning | 2 days | All features |

### Phase 4: Multi-User & Audio (Weeks 10-12)

**Goal**: Collaborative sessions, spatial audio, and session recording.

| Task | Duration | Dependencies |
|------|----------|-------------|
| Set up Yjs shared state | 2 days | Phase 3 |
| Implement WebRTC avatar sync | 3 days | Yjs |
| Implement avatar rendering | 2 days | Avatar sync |
| Implement spatial audio (PositionalAudio) | 2 days | Phase 3 |
| Implement attention sonification | 2 days | Spatial audio |
| Implement session recording | 3 days | Phase 3 |
| Implement session playback | 3 days | Session recording |

### Phase 5: Platform Integration & Polish (Weeks 13-16)

**Goal**: Full integration with Django, deployment, and final polish.

| Task | Duration | Dependencies |
|------|----------|-------------|
| Build VRRecording Django model + API | 2 days | Phase 4 |
| Build session management API | 2 days | VRRecording |
| Implement session sharing & embedding | 2 days | Session management |
| Implement collaborative annotation persistence | 2 days | Django integration |
| Apple Vision Pro testing and tuning | 3 days | Phase 4 |
| Cross-browser testing (Chrome, Firefox, Safari) | 3 days | All features |
| Documentation and tutorials | 3 days | All features |
| Performance optimization pass | 2 days | Testing results |

---

## Summary of Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rendering engine | React Three Fiber + Three.js | Largest ecosystem, best WebXR support, declarative API |
| GPU compute | WebGPU (WGSL) with WebGL2 fallback | 3-5x faster than WebGL; "10万粒子系统可达300FPS" |
| VR provider | @react-three/xr | Official Poimandres library, hand tracking support |
| Multi-user sync | Yjs CRDT + WebRTC DataChannel | Proven in collaborative 3D tools; P2P for low latency |
| Spatial audio | Three.js PositionalAudio + Web Audio API | Built-in HRTF, mature API, R3F integration |
| Session recording | Custom delta-encoded pose stream | Lightweight, replayable in the same scene |
| Platform backend | Django Channels (existing) | Reuses existing WebSocket infrastructure |
| Desktop fallback | OrbitControls + WASD | Same scene, no code duplication |

This specification provides a complete, implementable blueprint for the VR Immersion Room. Every component has been selected based on proven technology with strong community support and performance characteristics suitable for both desktop and standalone VR headsets. The phased implementation roadmap delivers incremental value, with each phase producing a usable feature set while building toward the full collaborative VR experience.