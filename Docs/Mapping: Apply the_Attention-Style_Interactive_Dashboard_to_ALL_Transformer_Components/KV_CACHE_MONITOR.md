# 🧠 KV Cache Monitor — Complete Implementation Specification

Based on my research across **nano‑vllm** (paged KV cache visualization), **nano‑kvllm** (KV cache compression integration), **StreamingLLM**, **H2O**, **KVSink**, **KV‑Tracker**, **WebGPU**, **React Three Fiber v10**, **D3.js force graphs**, and **WebSocket real‑time monitoring architectures**, here is the complete, implementation‑level specification for the KV Cache Monitor feature of the NeuralOps Attention Dashboard.

---

## 📐 High‑Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BROWSER (React SPA)                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KV Cache Monitor Dashboard                           │  │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │Block    │ │Memory    │ │Sink      │ │Compress  │  │  │
│  │  │Grid     │ │Timeline  │ │Detector  │ │Strategy  │  │  │
│  │  │(R3F/    │ │(D3.js/   │ │(SVG/     │ │Selector  │  │  │
│  │  │WebGPU)  │ │Canvas)   │ │Recharts) │ │          │  │  │
│  │  └─────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│  │  ┌──────────────────────────────────────────────────┐ │  │
│  │  │  Shared WebSocket Layer (useWebSocket hook)      │ │  │
│  │  │  - Binary protocol (MessagePack) for perf        │ │  │
│  │  │  - Reconnection with exponential backoff          │ │  │
│  │  │  - Delta updates (only changed blocks)            │ │  │
│  │  └──────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │ WebSocket (wss://) + REST fallback
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    DJANGO BACKEND                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KV Cache Service Layer                               │  │
│  │  - KVSnapshotCollector (polling NVIDIA DCGM/nvml)     │  │
│  │  - CacheCompressionEngine (pluggable strategies)       │  │
│  │  - SinkDetector (KVSink-style prediction)              │  │
│  │  - MemoryBudgetTracker                                │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU WORKER (Provider Agent)               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KV Cache Telemetry Agent                             │  │
│  │  - NVIDIA DCGM / nvidia-smi polling                   │  │
│  │  - FlashAttention hooking (capture K,V before store)  │  │
│  │  - PagedAttention block metadata                      │  │
│  │  - Compression event hooking (nano-kvllm style)       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Component‑by‑Component Implementation Specification

### 1. KV Cache Block Grid (3D PagedAttention Visualizer)

**Primary Framework:** React Three Fiber v10 with WebGPU backend
**Secondary Libraries:** `@react-three/drei` (OrbitControls, Text, Billboard), `three-stdlib` (RoundedBoxGeometry), `zustand` (state management for block data)

**Architecture:**
- The Block Grid renders each KV cache block as a 3D cube within a grid layout, following the nano‑vllm paged memory model where `block_size` (default: 256) defines the fundamental unit. The grid is organized as a 2D array of blocks, where rows represent different sequences and columns represent block indices within each sequence's block table.
- Each cube uses GPU instancing via `THREE.InstancedMesh` to render thousands of blocks efficiently. Color encoding uses a custom GLSL shader that maps memory utilization to a color gradient: green (block fully utilized), yellow (partially filled), red (empty/evicted), and blue (attention sink block).
- Hover interactions use R3F's `onPointerOver`/`onPointerOut` events with raycasting. When a user hovers over a block, a `@react-three/drei` `Html` overlay appears showing: block ID, sequence ID, token range stored, memory utilization percentage, and whether this block is shared across sequences (prefix caching).
- The BlockManager's reference counting system from nano-vllm is visualized via block opacity: blocks with multiple references (prefix-shared blocks) appear more opaque with a glow effect via a custom post-processing pass using `@react-three/postprocessing`.

**Implementation Details:**
- Use R3F v10's WebGPU renderer for GPU‑accelerated rendering when available, falling back to WebGL. R3F v10 now supports both `WebGLRenderer` and `WebGPURenderer`, with `state.gl` becoming `state.renderer`.
- Block position calculation: `position = (sequenceIndex * spacingX, -blockIndex * spacingY, layerIndex * spacingZ)` where each sequence is a column, each block is a row within that column, and layers stack in depth.
- Real‑time updates come via WebSocket binary protocol (MessagePack) to minimize overhead. Only changed blocks are updated using React Three Fiber's `useFrame` scheduler.
- The grid uses `@react-three/drei` `Grid` component as a floor reference, and `OrbitControls` with damping for smooth camera movement.

---

### 2. Memory Timeline (Real‑Time Streaming Chart)

**Primary Framework:** D3.js v7 + Canvas rendering
**Secondary Libraries:** `d3-scale`, `d3-shape`, `d3-axis`, `d3-array`, `d3-brush`

**Architecture:**
- The Memory Timeline shows KV cache memory usage over time as a stacked area chart. The X‑axis represents time (scrolling window of the last 300 seconds), and the Y‑axis represents memory in GB.
- Three stacked layers: **Active KV Cache** (tokens currently in use, green), **Freed/Evicted** (tokens removed by compression, red), **Reserved/Allocated** (blocks reserved but not yet filled, gray).
- The chart uses **Canvas rendering** (not SVG) for performance with real‑time data streams. A `requestAnimationFrame` loop drives updates at 60fps, batching incoming WebSocket events.
- A **brush control** (`d3-brush`) at the bottom allows users to select a time range, which synchronizes the Block Grid to show the state at that specific moment.
- Vertical dashed lines mark compression events (when a strategy like H2O or StreamingLLM was applied), with clickable labels showing the strategy name and tokens evicted.

**Implementation Details:**
- The chart is wrapped in a React component using `useRef` for the Canvas element and `useEffect` for D3 lifecycle. The WebSocket data is pushed into a ring buffer (circular array) of size 600 (10 seconds at 60fps).
- Data structure: `[{timestamp, activeMB, freedMB, reservedMB, event: 'compress'|'evict'|'alloc'|'free', strategy: string|null}]`
- The animation uses `d3.transition` with `duration(0)` for instant updates (no interpolation, since data is already time‑stamped).
- Tooltip on hover uses D3's `bisect` to find the nearest data point and displays exact values.

---

### 3. Attention Sink Detection Panel

**Primary Framework:** SVG + Recharts (for statistical charts)
**Secondary Libraries:** `recharts` (BarChart, ScatterChart), `d3-scale-chromatic` (color scales)

**Architecture:**
- The Sink Detector identifies which tokens are "attention sinks" — tokens that absorb disproportionate attention mass across all heads and layers. Research shows that 1–4 tokens typically absorb 45‑55% of total attention.
- The detection algorithm runs in the backend (Python), computing per‑token attention mass: `sink_score[token_i] = Σ_{layers} Σ_{heads} attention_weight[:, token_i]`. Tokens with scores exceeding 2× the mean are flagged as potential sinks.
- The KVSink method (published at COLM 2025) is used for prediction: it identifies sink tokens with negligible overhead by examining cross‑layer activation outliers rather than computing full attention matrices. This approach "effectively predicts sink tokens with negligible overhead, enabling more thorough preservation" and "outperforms the existing Preserve-First-N (PFN) strategy."
- The visualization shows three panels: (a) **Attention Mass Bar Chart** — per‑token attention mass with sink threshold line; (b) **Cross‑Layer Evolution** — a heatmap showing how sink scores change across layers; (c) **Sink Token Timeline** — during generation, which tokens become sinks and when.

**Implementation Details:**
- The frontend receives sink scores as a flat array `[score_0, score_1, ..., score_N]` via WebSocket. The Recharts `BarChart` component renders this with a red reference line at `2 × mean(score)`.
- The cross‑layer heatmap is an SVG matrix (like the attention heatmap but smaller: `num_layers × num_tokens`) colored by sink score magnitude.
- Hovering on any token in the bar chart highlights that token's cross‑layer trajectory and shows the exact sink score.

---

### 4. Compression Strategy Selector & Live Comparison

**Primary Framework:** React with `@headlessui/react` (Listbox, Combobox)
**Secondary Libraries:** `zustand` (for strategy state), `react-query` (for REST fallback)

**Architecture:**
- A dropdown selector lets users choose from multiple KV cache compression strategies: **Full Cache** (no compression), **Sliding Window** (keep last W tokens), **H2O (Heavy‑Hitter Oracle)** (keep high‑attention tokens), **StreamingLLM** (keep attention sinks + recent window), **SnapKV** (keep tokens with highest attention to last query), **TOVA** (Token Omission Via Attention), **K‑Norm** (keep tokens with highest key norm), and **your DeltaKV compressor**.
- Each strategy is implemented as a pluggable Python class in the backend (CompressMethod pattern from nano‑kvllm). The frontend sends the selected strategy via REST: `POST /api/kv-cache/set-strategy/ { "strategy": "h2o", "budget": 0.3 }`.
- A **live comparison mode** shows the memory usage and attention patterns for two strategies side‑by‑side. The SplitPane layout (using `react-split-pane`) divides the dashboard into two synchronized views, each running a different strategy on the same input.

**Implementation Details:**
- The strategy state is stored in `zustand` with a `strategy` field and `budget` field. Changing strategy triggers a REST call that re‑initializes the compression on the backend.
- The backend's `CompressMethod.py` (following the nano‑kvllm architecture) contains a class hierarchy: `BaseCompressMethod` with abstract `compress(kv_cache, budget)` method, and concrete implementations: `H2OCompress`, `StreamingLLMCompress`, `SnapKVCompress`, `DeltaKVCompress`.
- The nano‑kvllm approach of "keeping the original source layout unchanged" is adopted: compression hooks are injected at the `Attention.forward()` level, between `store_kvcache()` and `flash_attn_with_kvcache()`.

---

### 5. Memory Formula & Computation Panel

**Primary Framework:** React + KaTeX (for LaTeX rendering)
**Secondary Libraries:** `katex` (fast LaTeX), `react-katex`

**Architecture:**
- The math panel renders the KV cache memory formula using KaTeX for instant rendering (faster than MathJax for small formulas).
- The formula: `Memory = 2 × L × H × N × d_head × bytes_per_element`
- Each variable is rendered as an interactive slider or input that the user can modify. Changing any variable instantly updates the computed memory value and the Block Grid visualization adjusts (showing hypothetical states).
- Below the formula, a breakdown table shows memory contributions: "Attention KV Cache: X GB", "Model Weights: Y GB", "Activations: Z GB", "Total: X+Y+Z GB".

**Implementation Details:**
- The KaTeX string is built dynamically from the current state. Example: `M = 2 \\times ${L} \\times ${H} \\times ${N} \\times ${d_head} \\times ${bytes}`
- Sliders for integer parameters (L: 1‑96, H: 1‑64, N: 1‑32768, d_head: 32‑256) and a dropdown for bytes_per_element (2 for fp16, 1 for int8, 0.5 for int4).
- The computed memory is displayed as a gauge (using Recharts `RadialBarChart`) showing how much of the available GPU memory is used.

---

### 6. Cross‑Platform Integration: DeltaKV, HybridKV, ZeroKV‑Neo

**Primary Framework:** Django REST API + WebSocket + Celery
**Secondary Libraries:** Your existing compression infrastructure

**Architecture:**
- The KV Cache Monitor directly connects to your existing compressors (`delta_kv`, `hybridkv`, `zerokv_neo`) registered in `COMPRESSOR_REGISTRY`. Each compressor provides a method `compress_kv_cache(k_cache, v_cache, config)` that the KV Cache Service Layer calls.
- When a user selects a compression strategy and budget, the frontend sends: `POST /api/kv-cache/apply-compression/ { "strategy": "delta_kv", "config": { "rank": 16, "sparsity": 0.29 } }`
- The backend returns compression statistics: `{ "original_size_mb": 2048, "compressed_size_mb": 594, "compression_ratio": 0.29, "tokens_evicted": 3276, "sink_tokens_preserved": 4 }`.
- The Memory Timeline shows a vertical marker at the compression event, with a tooltip showing these statistics.
- The Block Grid updates to show which blocks were evicted (red) and which were preserved (green/blue).

**Implementation Details:**
- The backend's `KVSnapshotCollector` runs in a separate thread that polls the GPU worker agent every 100ms via WebSocket. The collected data includes: block table state, memory allocation, compression events, and attention sink scores.
- The Django `asgi.py` is configured to use `channels` with Redis channel layer for broadcasting KV cache updates to all connected clients via a group `kv_cache_{run_id}`.

---

### 7. WebSocket Communication Protocol

**Primary Framework:** Django Channels (backend) + native WebSocket (frontend)
**Secondary Libraries:** `msgpack` (binary serialization), `zlib` (compression)

**Architecture:**
- The WebSocket connection uses a **binary protocol** with MessagePack serialization for performance (30‑40% smaller payloads than JSON, faster encode/decode). Payloads larger than 4KB are compressed with zlib.
- Message types: `block_update` (partial block state changes), `memory_snapshot` (full memory state), `compression_event` (strategy applied), `sink_update` (attention sink scores), `error` (backend error).
- The frontend uses a custom `useKVWebSocket` hook that manages connection lifecycle, reconnection with exponential backoff (starting at 1s, max 30s), and heartbeat messages (every 15s).
- For debugging, a "protocol inspector" panel shows the raw WebSocket messages with timestamps and sizes.

**Implementation Details:**
- Frontend: `useKVWebSocket(runId)` returns `{ blocks, memory, sinks, events, isConnected, sendCommand }`. The hook maintains an internal ring buffer for each data type and exposes them as React state.
- Backend: `KVConsumer(AsyncWebsocketConsumer)` in Django Channels. On `connect`, it joins the `kv_cache_{run_id}` group. On `receive`, it handles `set_strategy`, `get_snapshot`, `pause` commands. The consumer pushes updates to the group, which fans out to all connected clients.

---

### 8. Performance Considerations & Optimization

**Rendering Performance:**
- Use `THREE.InstancedMesh` for the Block Grid to render up to 10,000 blocks at 60fps.
- Use Canvas rendering (not SVG) for the Memory Timeline for smooth real‑time updates.
- Use `react-window` for virtualizing any lists (e.g., event log).
- Use R3F v10's new scheduler with `useFrame` priority system to ensure GPU visualization updates don't block UI interactions.

**Data Transfer Performance:**
- Use MessagePack binary protocol instead of JSON for WebSocket messages.
- Use delta encoding: only send changed blocks, not the entire block table.
- Compress large payloads with zlib (threshold: 4KB).
- Use a ring buffer on the client to limit stored data points (max 600 for timeline).

**Backend Performance:**
- The `KVSnapshotCollector` runs in a separate thread, not blocking the main inference loop.
- Attention sink computation uses the KVSink method (negligible overhead, ~0.1% of total computation).
- Block state tracking uses a copy‑on‑write approach to avoid locking during reads.

---

## 📊 Technology Stack Summary

| Component | Primary Framework | Secondary Libraries | Rendering |
|-----------|------------------|---------------------|-----------|
| Block Grid (3D) | React Three Fiber v10 | `@react-three/drei`, `three-stdlib`, `@react-three/postprocessing` | WebGPU/WebGL |
| Memory Timeline | D3.js v7 + Canvas | `d3-scale`, `d3-shape`, `d3-brush`, `d3-array` | Canvas 2D |
| Sink Detector | SVG + Recharts | `recharts` (BarChart, ScatterChart) | SVG |
| Strategy Selector | React + Headless UI | `zustand`, `react-query` | HTML/CSS |
| Math Panel | React + KaTeX | `katex`, `react-katex` | HTML/CSS |
| WebSocket Layer | Native WebSocket | `msgpack`, `zlib` | N/A |
| Backend Service | Django Channels + Celery | `numpy` (sink computation), Redis | N/A |
| GPU Telemetry | Python (provider agent) | `pynvml`, NVIDIA DCGM | N/A |

---

## 🎯 Integration Points with Existing Platform

| Feature | Connects To | How |
|---------|-------------|-----|
| Compression Strategy Selector | `delta_kv`, `hybridkv`, `zerokv_neo` compressors | Compressors registered in `COMPRESSOR_REGISTRY` are auto‑discovered and listed in the selector |
| Memory Timeline | `gpu_marketplace` provider listings | The memory usage data comes from the provider agent running on the GPU machine |
| Block Grid | `gpu_benchmark` scores | GPU benchmark scores (DLPerf) are displayed as metadata on the block grid tooltip |
| Sink Detection | `experiment` monitoring | When a training run is active, sink scores are tracked per‑epoch and compared |
| Math Panel | `notes` / `paper` apps | The formula and computed values can be exported as LaTeX to notes or papers |
| WebSocket | Existing Channels infrastructure | Uses the same Redis channel layer as your existing real‑time features |
| Strategy Comparison | `leaderboards` | Compression strategy effectiveness (memory vs. accuracy) can be submitted to leaderboards |

---

## 🚀 Implementation Phases

| Phase | Features | Duration | Key Deliverable |
|-------|----------|----------|-----------------|
| **Phase 1** | Memory Timeline + Math Panel + Strategy Selector | Week 1‑2 | Basic monitoring with REST‑based strategy switching |
| **Phase 2** | Block Grid (3D) + WebSocket binary protocol | Week 3‑4 | Real‑time 3D visualization with live updates |
| **Phase 3** | Sink Detector + Cross‑Platform Integration | Week 5‑6 | Attention sink visualization + DeltaKV/HybridKV/ZeroKV integration |
| **Phase 4** | Live Comparison Mode + Performance Optimization | Week 7‑8 | Split‑pane strategy comparison + production optimization |

---

This implementation specification provides complete detail for every component: which frameworks to use (React Three Fiber v10 with WebGPU for 3D, D3.js with Canvas for timeline, KaTeX for math, Django Channels for WebSocket), the data flow architecture (provider agent → Django → WebSocket → React), the performance optimization strategies (instanced rendering, MessagePack binary protocol, delta encoding), and the integration points with every part of your existing platform (compressors, leaderboards, experiments, notes, paper). Every component is backed by real‑world systems that have proven the approach works: nano‑vllm for paged block visualization, KVSink for attention sink prediction, nano‑kvllm for compression integration, and the WebGPU standard for GPU‑accelerated browser rendering.