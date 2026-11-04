# NeuralOps 3D Neural Architecture Visualizer

NeuralOps includes a high-performance WebGL / Three.js 3D architecture visualizer to interactively inspect, animate, and compare 131 attention mechanisms and 72 model compression topologies.

The visualizer provides spatial token routing graphs, receptive field projections, and low-rank singular value spectrums.

---

## Interactive Cloud Playground

The interactive web demo is hosted live on Hugging Face Spaces:

🔗 **Live Space:** [https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer](https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer)

### Key Features:
- **Spatial Routing Visualizer**: Visualize quadratic ($O(L^2)$) vs linear ($O(L)$) state propagation across transformer layers.
- **Spectrum Explorer**: Real-time singular value decay plots for decomposed weight matrices.
- **Multi-Topology Compare**: Side-by-side comparison of baseline Multi-Head Attention vs surgically transplanted linear/recurrent attention.
- **Dynamic Head Pruning**: Interactive visualization of head sparsity patterns under 2:4 structured sparsity.

---

## Running Locally

The visualizer can be served directly from local source without requiring external build dependencies.

### Serving with Python HTTP Server:

```bash
# From the repository root:
python -m http.server 8000 --directory docs/visualizer
```

Then open your browser at:
```
http://localhost:8000/index.html
```

Or open `standalone.html` for offline self-contained inspection:
```bash
open docs/visualizer/standalone.html
```

---

## CLI Integration

NeuralOps CLI can launch the visualizer directly:

```bash
# Open interactive 3D visualizer in your default browser
neuralops visualizer --port 8000
```
