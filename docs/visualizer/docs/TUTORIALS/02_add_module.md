# Tutorial 02: Adding a Custom Attention Module

In this tutorial, we will create and register a new low-rank attention module named `HyperAttention`.

---

## Step 1: Create the Component JSON
Create `Docs/visualizer/components/hyper_attention.json`:

```json
{
  "id": "hyper_attention",
  "name": "HyperAttention: Kernelized Subspace Sampling",
  "file": "Codes/framework/attentions/hyper_attention.py",
  "type": "Attention Module",
  "category": "Dynamic & Efficient Attention",
  "paradigm": "svd",
  "icon": "⚡",
  "description": "Applies randomized Johnson-Lindenstrauss projections to approximate attention in near-linear time.",
  "math_latex": "A \\approx \\text{Softmax}(Q \\Omega (K \\Omega)^\\top) V",
  "components": [
    {
      "id": "jl_proj",
      "name": "Johnson-Lindenstrauss Random Projection",
      "role": "Dimensionality Reduction",
      "shape_transform": "[m \\times d]",
      "color": "#38bdf8"
    },
    {
      "id": "kernel_eval",
      "name": "Fast Kernelized Evaluator",
      "role": "Attention Approximation",
      "shape_transform": "[d \\times d]",
      "color": "#2dd4bf"
    }
  ],
  "model_projections": {
    "distilbert": {
      "model_name": "DistilBERT-Base",
      "savings_ratio": 3.8,
      "storage_saved_formatted": "38.5 MB",
      "benchmark": { "status": "PASS", "accuracy_pct": 81.9 }
    }
  }
}
```

---

## Step 2: Regenerate Catalog Index
Run:
```bash
python3 Docs/visualizer/export_module_catalog.py
```

---

## Step 3: Test in Visualizer
1. Open `Docs/visualizer/index.html`.
2. Open the Library Modal (`📚 Library`).
3. Search for `HyperAttention`.
4. Click **Apply to 3D**. The right column will immediately re-render using the `HyperAttention` component topology!
