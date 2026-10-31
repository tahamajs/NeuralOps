# Extending the 3D Neural Architecture Visualizer

This guide explains how to add new attention modules, compression algorithms, or foundation models.

---

## 1. Adding a New Compression or Attention Module

### Step 1: Place Module in Framework
Save your PyTorch implementation in either:
* `Codes/framework/attentions/<my_module>.py`
* `Codes/framework/compression_engine/<my_module>.py`

### Step 2: Define Component Topology in `components/<my_module>.json`
Create `Docs/visualizer/components/<my_module>.json`:

```json
{
  "id": "my_module",
  "name": "My Efficient Attention",
  "file": "Codes/framework/attentions/my_module.py",
  "type": "Attention Module",
  "category": "Low-Rank SVD & Factorization",
  "paradigm": "svd",
  "icon": "⚡",
  "description": "Factorizes QKV with custom rank bottleneck.",
  "math_latex": "W_{proj} = U \\cdot V, \\quad r=32",
  "components": [
    {
      "id": "v_factor",
      "name": "Factor V (Subspace Proj, r=32)",
      "role": "Rank-32 Compression",
      "shape_transform": "[32 × d_in]",
      "color": "#3acbb8"
    },
    {
      "id": "u_factor",
      "name": "Factor U (Expansion, r=32)",
      "role": "Reconstruction",
      "shape_transform": "[d_out × 32]",
      "color": "#2dd4bf"
    }
  ],
  "model_projections": {
    "distilbert": {
      "model_name": "DistilBERT-Base",
      "savings_ratio": 4.5,
      "storage_saved_formatted": "45.2 MB",
      "benchmark": { "status": "PASS", "accuracy_pct": 82.5 }
    }
  }
}
```

### Step 3: Re-index Catalog
Run the catalog generator:
```bash
python3 Docs/visualizer/export_module_catalog.py
```
Your module will immediately appear in the Library Modal and Quick Pick dropdown!

---

## 2. Adding a New Model Architecture

1. **Add Specs to `src/constants.js`**:
   ```javascript
   MODEL_SPECS.my_model = {
       name: 'MyModel-7B',
       family: 'Decoder',
       num_layers: 32,
       hidden_size: 4096,
       intermediate_size: 11008,
       base_layer_params: 200000000,
       base_total: 6700000000,
       descr: '32 Layers · 4096d · 32 Heads',
       vocab_size: 32000
   };
   ```
2. **Add Branch to `src/physics/token_trajectory.js`**:
   Add custom particle routing for any non-standard tensor splits (e.g. cross-attention, recurrent states).
3. **Run `python3 Docs/visualizer/generate_module_components.py`** to recalculate projections across all 240+ modules.
