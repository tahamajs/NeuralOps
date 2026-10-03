# Data Schema Documentation · 3D Neural Architecture Visualizer

This document formally specifies all data schemas used by the 3D Neural Architecture Visualizer (`Docs/visualizer`).
The visualizer consumes three primary data files and a directory of modular JSON component topologies:

1. [`spatial-data.json`](Docs/visualizer/spatial-data.json) (~3.1 MB)
2. [`modules-catalog.json`](Docs/visualizer/modules-catalog.json) (~543 KB)
3. [`modules_components.json`](Docs/visualizer/modules_components.json) (~1.4 MB)
4. [`components/<module_id>.json`](Docs/visualizer/components) (240 files)

---

## 1. `spatial-data.json` Schema

`spatial-data.json` provides the spatial scene graph, node bounding boxes, 3D coordinates, and structural conduits across all 4 supported model families.

```json
{
  "title": "3D Neural Architecture Spatial Layout",
  "generated_at": "ISO-8601 string",
  "models": {
    "distilbert": ModelSpatialData,
    "roberta": ModelSpatialData,
    "gpt2": ModelSpatialData,
    "qwen": ModelSpatialData
  }
}
```

### `ModelSpatialData` Object
| Field | Type | Description |
|---|---|---|
| `true_num_layers` | `number` | Physical number of transformer layers in model (e.g. 6, 12, 24) |
| `capped_num_layers` | `number` | Rendered layers in the 3D canvas (capped for 60fps WebGL rendering) |
| `baseline_nodes` | `Node3D[]` | Array of dense uncompressed nodes (left column, $X = -4.5$) |
| `baseline_macro_nodes` | `MacroNode[]` | Consolidated macro blocks for far-distance LOD 0 |
| `compressed_nodes` | `Node3D[]` | Array of default SVD-compressed nodes (right column, $X = +4.5$) |
| `macro_nodes` | `MacroNode[]` | Compressed macro blocks for LOD 0 |
| `edges` | `Edge3D[]` | Structural connecting curves and conduits between tensor nodes |
| `max_y` | `number` | Maximum vertical scene height in Three.js world units |
| `num_baseline` | `number` | Count of baseline nodes |
| `num_compressed` | `number` | Count of compressed nodes |
| `total_baseline_params` | `number` | Exact baseline parameter integer |
| `total_baseline_params_formatted` | `string` | Formatted parameter string (e.g. `"66.36M"`) |
| `total_compressed_params` | `number` | Exact compressed parameter integer |
| `total_compressed_params_formatted` | `string` | Formatted compressed parameter string (e.g. `"29.20M"`) |
| `overall_compression_ratio` | `number` | Reduction factor (e.g. `2.27`) |
| `storage_saved_bytes` | `number` | Storage delta in bytes |
| `storage_saved_formatted` | `string` | Formatted storage savings (e.g. `"148.6 MB"`) |
| `modules` | `Record<string, ModuleSpatialData>` | Embedded spatial layouts for default paradigms (`svd`, `moe`, `mla`, `slicegpt`) |

### `Node3D` Schema
| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string` | **YES** | Unique identifier (e.g. `"layer.0.attention.q_lin"`) |
| `type` | `string` | **YES** | Functional category (`"dense"`, `"svd"`, `"moe"`, `"mla"`, `"embedding"`, `"output_head"`, `"residual_add"`) |
| `sub_type` | `string` | NO | Sublayer specialization (`"v_factor"`, `"u_factor"`, `"router"`, `"expert"`, `"gate_proj"`) |
| `parent_layer_index` | `number` | **YES** | Zero-indexed transformer layer index |
| `parent_layer_name` | `string` | **YES** | Human-readable parent name (e.g. `"Layer 0"`) |
| `sequence_index` | `number` | **YES** | Execution sequence order within the layer |
| `x` | `number` | **YES** | Three.js X world position |
| `y` | `number` | **YES** | Three.js Y world position (vertical height) |
| `z` | `number` | **YES** | Three.js Z world position (depth plane) |
| `width` | `number` | **YES** | Tensor mesh box width (X-dimension) |
| `height` | `number` | **YES** | Tensor mesh box height (Y-dimension) |
| `depth` | `number` | **YES** | Tensor mesh box depth (Z-dimension) |
| `label` | `string` | **YES** | Display title rendered on billboard sprite badges |
| `metadata` | `NodeMetadata` | **YES** | Detailed shape, parameter counts, and theoretical storage |

### `NodeMetadata` Schema
```typescript
interface NodeMetadata {
  shape: [number, number] | [number]; // Matrix dimensions [d_out, d_in]
  params: string;                     // e.g. "590.6K" or "23.44M"
  params_raw: number;                 // Integer parameter count (e.g. 590592)
  storage: string;                    // Estimated FP32/FP16 footprint (e.g. "2.36 MB")
  storage_bytes: number;              // Raw storage footprint in bytes
  dense_equivalent_params?: number;   // Baseline parameters before compression
  dense_equivalent_params_formatted?: string;
  compression_ratio?: number;         // Ratio relative to dense baseline (1.0 = baseline)
  compression?: string;               // Formatted string (e.g. "2.27x (-56.0%)")
  type?: string;                      // Human description (e.g. "Dense Linear Projection")
  rank?: number;                      // Low-rank bottleneck rank r (for SVD layers)
  heads?: number;                     // Attention head count (for MHA/GQA projections)
  num_experts?: number;               // Expert count (for MoE layers)
}
```

---

## 2. `modules-catalog.json` Schema

`modules-catalog.json` catalogs all 240+ research attention modules and compression engines in the repository, indexed for fast filtering, search, and preview in the Library Modal.

```json
{
  "total_count": 240,
  "generated_at": "ISO-8601 string",
  "modules": [ModuleCatalogEntry]
}
```

### `ModuleCatalogEntry` Schema
| Field | Type | Description |
|---|---|---|
| `id` | `string` | Canonical module identifier matching codebase filename (e.g. `"SOAFusedAttention"`, `"a3_compressor"`) |
| `name` | `string` | Formatted display title |
| `file` | `string` | Relative path to implementation in `Codes/framework/` |
| `type` | `string` | `"Attention Module"` or `"Compression Engine"` |
| `category` | `string` | One of 7 primary research categories (see `PARADIGMS` in `constants.js`) |
| `paradigm` | `string` | Primary algorithmic paradigm (`"svd"`, `"moe"`, `"ssm"`, `"mla"`, `"slicegpt"`, `"quant"`) |
| `icon` | `string` | UI emoji indicator (e.g. `"⚡"`, `"🔀"`, `"🐍"`, `"💎"`) |
| `description` | `string` | Extracted module docstring / architectural summary |
| `math` | `string` | Symbolic mathematical formulation (LaTeX / plain-text) |
| `markdown_preview` | `string` | Markdown excerpt from canonical documentation in `Docs/content/` |
| `verified_benchmark` | `BenchmarkResults` | Benchmark scores harvested from `results/` |

---

## 3. `modules_components.json` Schema

`modules_components.json` provides the complete deep-inspection specification for every module, including its sublayer decomposition, before-and-after structural transformation rules, and cross-model parameter projections across all 4 architectures.

```json
{
  "total": 240,
  "modules": [ModuleComponentDefinition]
}
```

### `ModuleComponentDefinition` Schema
| Field | Type | Description |
|---|---|---|
| `id` | `string` | Canonical module identifier |
| `name` | `string` | Human-readable module name |
| `file` | `string` | Source file path in `Codes/framework/` |
| `type` | `string` | Component type |
| `category` | `string` | Architectural category |
| `paradigm` | `string` | Algorithmic paradigm key |
| `icon` | `string` | Visual indicator icon |
| `description` | `string` | Detailed technical description |
| `classes` | `string[]` | Python class names defined in the module |
| `methods_count` | `number` | Number of methods in the main module class |
| `math_latex` | `string` | Complete LaTeX equation for KaTeX rendering |
| `components` | `SublayerComponent[]` | Breakdown of internal operator sublayers (e.g. Factor V, Factor U) |
| `model_projections` | `Record<ModelKey, ModelProjection>` | Model-specific projections across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 |
| `verified_benchmark` | `BenchmarkResults` | Verified empirical accuracy, throughput, and memory stats |

### `SublayerComponent` Schema
```typescript
interface SublayerComponent {
  id: string;              // e.g. "v_factor", "u_factor", "router", "expert"
  name: string;            // e.g. "Factor V (Low-Rank Subspace Projection, r=64)"
  role: string;            // e.g. "Projection to Rank-r Eigenspace"
  shape_transform: string; // e.g. "[64 × d_in]", "[d_out × 64]"
  color: string;           // Hex color code (e.g. "#3acbb8")
}
```

### `ModelProjection` Schema
```typescript
interface ModelProjection {
  model_name: string;               // e.g. "DistilBERT-Base", "Qwen2.5-0.5B"
  family: string;                   // "Encoder", "Causal Decoder", "GQA Decoder (SLM)"
  hidden_size: number;              // Hidden dimension d_model (768 or 896)
  intermediate_size: number;        // MLP intermediate dimension (3072 or 4864)
  num_layers: number;               // Number of layers (6, 12, or 24)
  baseline_layer_params: number;    // Params in one uncompressed baseline layer
  compressed_layer_p: number;       // Params in one compressed layer
  total_baseline_params: number;    // Full uncompressed model param count
  total_compressed_params: number;  // Full compressed model param count
  savings_ratio: number;            // Savings ratio (e.g. 2.27, 8.0)
  storage_saved_bytes: number;      // Storage reduction in bytes
  storage_saved_formatted: string;  // Formatted string (e.g. "70.9 MB")
  benchmark?: BenchmarkResults;     // Empirical validation metrics
}
```

### `BenchmarkResults` Schema
```typescript
interface BenchmarkResults {
  status: "PASS" | "PASS_LOW" | "FAIL" | "SKIP" | "PENDING";
  accuracy_pct?: number;           // Downstream GLUE accuracy (e.g. 81.31%)
  f1_score?: number;               // F1 score
  peak_vram_mb?: number;           // Peak GPU memory in MB
  throughput_tokens_sec?: number;  // Inference throughput in tokens/sec
  source_file?: string;            // Path to CSV log in results/
}
```

---

## 4. `components/<module_id>.json` Schema

Each of the 240 standalone files inside `Docs/visualizer/components/` is an exact single-module export of the `ModuleComponentDefinition` schema above. This allows modular on-demand fetching when selecting any research module without parsing the full consolidated catalog upfront.
