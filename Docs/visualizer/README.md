# 3D Neural Architecture Visualizer · Multi-Model Atlas

A high-performance, interactive 3D WebGL digital twin for visualizing and inspecting dense transformer models alongside their compressed low-rank (SVD) and sparse mixture-of-experts (MoE) counterparts.

---

## 🚀 Quickstart (How to Run)

Choose any of the three zero-setup options below:

### Option 1 — One-Click Automated Server (Recommended)
```bash
python3 Docs/visualizer/serve.py
```
*Automatically finds an open port (starting at 8000), avoids conflicts, launches your default web browser, and stops cleanly with `Ctrl+C`.*

### Option 2 — Raw Python HTTP Server
```bash
python3 -m http.server 8000 --directory Docs/visualizer
```
*Then visit [http://localhost:8000/index.html](http://localhost:8000/index.html) in any modern browser.*

### Option 3 — 100% Offline Standalone File (No Server Needed)
```bash
open Docs/visualizer/standalone.html
```
*Or double-click `standalone.html` directly in macOS Finder or your operating system's file manager.*

---

## 🧠 Educational Value: What the Visualizer Teaches

This tool is designed for ML researchers, systems engineers, and students studying model compression techniques. The dual synchronized viewport visualizes forward-pass activation dynamics side-by-side:
- **Left Viewport (Dense Baseline):** Renders standard full-rank matrix projections ($O(d_{\text{in}} \times d_{\text{out}})$). Tokens flow as broad parallel streams through full geometric tensors.
- **Right Viewport (Compressed Model):** Renders factorized low-rank architectures ($W \approx U \cdot V$) with bottleneck rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$.
- **Physical Token Dynamics:** Flowing tokens physically pinch inward and experience simulated velocity slowdown through the thin $V$-factor bottleneck before expanding back through the $U$-factor. In MoE blocks, tokens branch into distinct color-coded trajectories through parallel expert modules.
- **Hierarchical Inspection:** The collapsible Layer Explorer drawer allows selecting any transformer block or projection to view exact dimensional shapes, parameter counts, memory storage, and real parameter savings without guessing.

---

## 🛠 Project Structure

```
Docs/visualizer/
├── model_to_3d.py       # Config-only PyTorch exporter generating spatial-data.json
├── spatial-data.json    # Pre-calculated 3D bounding geometry & compression metrics
├── index.html           # Split-screen UI, layer tree drawer, and global comparison HUD
├── style.css            # Futuristic glassmorphic styling, neon edge glows & animations
├── app.js               # Three.js dual-viewport renderer, token physics & interactions
├── serve.py             # Resilient local server launcher with auto port-finding
├── build_standalone.py  # Compiler that inlines all assets into standalone.html
├── standalone.html      # Fully offline single-file distribution (zero network requests)
└── README.md            # Documentation & usage guide
```

---

## ⚠️ Known Limitations & Design Trade-offs

1. **Proxy Models for Architecture Topologies:**
   - Qwen uses `Qwen/Qwen2.5-0.5B` architecture configuration to structurally represent the Qwen3.5 family (GQA, RoPE, SwiGLU).
   - DistilBERT (`distilbert-base-uncased`), RoBERTa (`roberta-base`), and GPT-2 (`gpt2`) instantiate their official config topologies.
2. **Layer Depth Capping:**
   - Deep models (e.g. 24-layer Qwen) have their spatial visualization capped at 12 layers for optimal WebGL frame-rate and viewport clarity. The un-truncated layer count is preserved in metadata (`true_num_layers`).
3. **Pure Structural Topology:**
   - The visualizer constructs model architectures purely from Hugging Face `AutoConfig` definitions and generates low-rank geometry mathematically. Full pre-trained model weights are never downloaded, ensuring instantaneous export and lightweight deployment.
