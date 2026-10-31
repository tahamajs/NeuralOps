# 🔬 Zone 6: Interpretability Tools — Complete Implementation Specification

## Overview

Zone 6 is the **deep inspection layer** of the attention dashboard. It provides four interlinked tools that let researchers answer the hardest questions about transformer models: *What does this head do? How do heads communicate? Which heads matter for a prediction? Can we trace the causal chain?* 

The four tools—Head Ablation Laboratory, Circuit Tracer, Dodrio‑Style Linguistic Overlay, and Causal Tracing—share a common backend architecture but each presents a radically different UX. They are built on **four core Python libraries** (TransformerLens, nnsight, pyvene, circuit‑tracer), **three JavaScript visualization engines** (D3.js, React‑Three‑Fiber, PixiJS), and **one high‑performance communication layer** (WebSocket + SharedArrayBuffer).

---

## Shared Architecture: The Intervention Backend

All four tools in Zone 6 share a single Python backend that wraps the target model and exposes intervention capabilities through a **WebSocket API**. This backend runs as a persistent process (or Celery task) that loads the model once, then accepts intervention commands from the frontend, executes them, and streams results back.

### Backend Stack: Five Python Libraries That Do Different Things

The backend uses **five libraries**, each solving a distinct part of the intervention pipeline. Understanding why each is needed—rather than what API it exposes—is key to the architecture.

**TransformerLens (the standard library)** provides `HookedTransformer`, a model wrapper that adds `HookPoint` objects after every sublayer (attention, MLP, residual). This lets you register callbacks on any intermediate activation without modifying the model code itself. It also includes built‑in activation patching utilities and supports 50+ open‑weights models. TransformerLens is used here primarily for **model loading and hook registration**—it provides the cleanest interface for attaching intervention points to GPT‑2, Pythia, Gemma, and Llama‑style architectures.

**nnsight (the deferred execution engine)** wraps any PyTorch model and provides a `model.trace()` context manager that captures intervention code before executing it. Unlike TransformerLens, nnsight works with **arbitrary PyTorch models** (not just GPT‑style), including vision transformers and custom architectures. Its key innovation is that you write intervention code *as if* the model were running, but nnsight defers execution, optimizes the computation graph, and runs it efficiently. For the ablation laboratory, nnsight is preferred because it can apply interventions persistently via `model.edit()`, meaning ablated heads stay ablated across multiple forward passes without re‑registering hooks.

**pyvene (the causal intervention framework)** is built by Stanford NLP and provides a declarative, dict‑based way to specify complex interventions. Instead of writing raw hook functions, you define an `IntervenableConfig` that describes *what* activations to patch, *where* to patch them from, and *how* to combine them. pyvene handles the bookkeeping of clean vs. corrupted runs, interchange interventions, and counterfactual analysis. It is the engine behind the Causal Tracing tool, where researchers need to systematically test which activations causally affect the output.

**circuit‑tracer (the attribution graph generator)** is Anthropic's open‑source library for computing attribution graphs from transcoder features. It takes a model with pre‑trained cross‑layer transcoders (CLTs), runs a forward pass, and computes the direct effect that each non‑zero transcoder feature has on every other feature and on the output logits. The result is a JSON graph where nodes are "supernodes" (clusters of semantically related features) and edges are weighted attribution scores. This is the engine behind the Circuit Tracer tool.

**spaCy + Stanza (the linguistic annotation pipeline)** provide tokenization, part‑of‑speech tagging, dependency parsing, named entity recognition, and coreference resolution. They run on the same input text that was fed to the model, producing structured annotations that can be overlaid on attention visualizations. spaCy is used for English (fast, production‑ready); Stanza provides coverage for 68 languages through the same interface.

### Model Loading Strategy: Three‑Tier Architecture

The backend loads models in a three‑tier architecture to minimize startup time and memory:

**Tier 1 — Cached Hooked Models.** For GPT‑2, Pythia, Gemma, and Llama architectures, the backend uses TransformerLens's `HookedTransformer.from_pretrained()`, which downloads model weights once and caches them. Subsequent loads are near‑instantaneous (under 200ms for GPT‑2‑small).

**Tier 2 — nnsight Wrapper.** For models not supported by TransformerLens (custom architectures, vision transformers, multimodal models), the backend wraps them with `nnsight.LanguageModel()` or `nnsight.NNsight()`. This provides the same intervention interface (`model.trace()`, `model.edit()`) regardless of the underlying architecture.

**Tier 3 — HuggingFace + Custom Hooks.** For models where neither TransformerLens nor nnsight provides the needed granularity, the backend loads the model via HuggingFace `AutoModel` and registers PyTorch forward hooks manually using `register_forward_hook()`. This is the most flexible but most labor‑intensive approach, reserved for cutting‑edge models.

The backend selects the tier automatically based on the model ID. The selection logic is:

```python
if model_id in TRANSFORMER_LENS_SUPPORTED:
    model = HookedTransformer.from_pretrained(model_id)
elif model_id in NNSIGHT_SUPPORTED:
    model = LanguageModel(model_id, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(model_id)
    register_custom_intervention_hooks(model)
```

### The Intervention WebSocket Protocol

The frontend communicates with the backend through a **bidirectional WebSocket** using a JSON‑based protocol. This is not a REST API because interventions need to be streaming and interactive—the user flips a toggle, the backend runs the model, and results stream back within 50–200ms.

**Message types from frontend to backend:**

| Message Type | Payload | Purpose |
|---|---|---|
| `load_model` | `{model_id, attention_type, attention_config}` | Load a new model (or attention variant) |
| `set_text` | `{text}` | Set the input text for all subsequent interventions |
| `ablate_head` | `{layer, head, enabled}` | Enable or disable a specific attention head |
| `ablate_chain` | `{heads: [{layer, head, enabled}]}` | Batch enable/disable multiple heads |
| `get_attribution_graph` | `{prompt}` | Generate a full attribution graph for the given prompt |
| `run_causal_trace` | `{source_token, target_token, layers}` | Run causal tracing between two tokens |
| `get_linguistic_annotations` | `{text}` | Get dependency parse, POS tags, coreference chains |
| `compare_ablations` | `{ablations: [{name, heads}]}` | Compare multiple ablation configurations |
| `get_intervention_summary` | — | Get the current intervention state and model info |

**Message types from backend to frontend:**

| Message Type | Payload | Purpose |
|---|---|---|
| `model_loaded` | `{model_id, layers, heads_per_layer, vocab_size}` | Confirmation that model is ready |
| `attention_computed` | `{tokens, layers: [{layer, heads: [{head, matrix}]}]}` | Full attention data for the current text |
| `ablation_result` | `{before_logits, after_logits, affected_tokens}` | Logit changes from the most recent ablation |
| `attribution_graph` | `{nodes: [{id, label, type, layer}], edges: [{source, target, weight}]}` | Full attribution graph |
| `causal_trace_result` | `{paths: [{tokens, scores}], critical_path: [...]}` | Causal tracing results |
| `linguistic_annotations` | `{tokens: [{text, pos, dep, head, ner}], dependencies: [{source, target, label}]}` | Linguistic annotation data |
| `intervention_state` | `{ablated_heads: [{layer, head}], model_id, text}` | Current full intervention state |
| `logit_lens` | `{layer_predictions: [{layer, top_k: [{token, prob}]}]}` | Logit lens data for all layers |
| `error` | `{code, message}` | Error notification |

### Performance Budget for Interventions

The intervention backend must meet strict latency targets to feel interactive:

| Operation | Target Latency | Implementation Strategy |
|---|---|---|
| Single head ablation (cached) | < 100ms | Run forward pass with one hook modification; cache clean activations |
| Chain ablation (5+ heads) | < 300ms | Run single forward pass with all hooks applied simultaneously |
| Attribution graph generation | < 5s | Use circuit‑tracer's optimized backend; run on GPU if available |
| Causal tracing (10 layers) | < 2s | Run patching in parallel across layers using nnsight's multi‑invoke |
| Linguistic annotation | < 200ms | Run spaCy pipeline; cache annotations per text |
| Logit lens (all layers) | < 1s | Single forward pass; collect logits at each layer via hooks |

---

## Feature 21: Head Ablation Laboratory — Implementation

### Purpose

The Head Ablation Laboratory lets researchers answer: *"What happens if I remove this head? What if I remove these three heads together? Which heads are critical, and which are redundant?"* The tool provides real‑time feedback: flip a toggle, see the model's output change.

### Backend Implementation: The Intervention Pipeline

The ablation backend uses nnsight's `model.edit()` for persistent interventions. Here is the architecture:

1. **Model wrapping**: The model is wrapped with `nnsight.LanguageModel(model_id, device_map="auto")`.
2. **Clean forward pass**: On text input, a clean forward pass is run once. All intermediate activations (attention patterns, MLP outputs, residual stream values) are cached in memory.
3. **Intervention application**: When the user toggles a head, the backend creates an intervention that zeros out the attention weights for that specific head. Using nnsight, this is:

```python
with model.edit("ablated"):
    # For each ablated head, set its attention weights to zero
    for layer, head in ablated_heads:
        # Access the attention pattern for this specific head
        hook_name = f"transformer.h.{layer}.attn.attn_pattern"
        def zero_head(pattern, head_idx=head):
            pattern[:, head_idx, :, :] = 0
            return pattern
        model.add_intervention(hook_name, zero_head)
```

4. **Edited forward pass**: The model runs with all ablations applied simultaneously. The output logits are compared with the clean run.
5. **Impact computation**: For each ablated head configuration, the backend computes:
   - **Logit difference**: How much did each token's probability change?
   - **KL divergence**: How different is the output distribution from the clean run?
   - **Top‑k flip**: Did the top prediction change?
6. **Result streaming**: Results are sent back via WebSocket as they are computed.

### Caching Strategy for Fast Toggle Response

When a user rapidly toggles heads on and off, recomputing the full forward pass each time would be wasteful. The backend uses a **cache‑and‑invalidate** strategy:

- **Clean cache**: The attention patterns and activations from the clean run are cached in GPU memory (or CPU if GPU memory is limited). This cache is invalidated only when the text changes.
- **Ablation overlay**: For each toggled head, the backend doesn't re‑run the entire model—it only recomputes the *affected layers*. Layers above the ablated head can reuse cached activations. Layers at and below the ablated head must be recomputed.
- **Chain ablation**: When multiple heads across different layers are ablated, the backend identifies the *lowest affected layer* and recomputes from there upward, applying all ablations in a single pass.

### Frontend Implementation: The Ablation Control Panel

The ablation panel is rendered as an **interactive grid** matching the Model View from Zone 2. Each head is a colored cell. A toggle switch on each cell enables/disables that head.

**Visual states for each cell:**

| State | Appearance | Meaning |
|---|---|---|
| Active (default) | Full color, normal opacity | Head is contributing normally |
| Ablated | Grayed out, 30% opacity, red X overlay | Head is disabled |
| Hovered | Bright border, scale 1.05 | Mouse is over this head |
| Ablated + Hovered | Gray with bright border | Head is disabled, user is inspecting it |
| Impact‑highlighted | Pulsing glow animation | This head was just toggled and had a large impact |

**Cell interactions:**
- **Click**: Toggle ablation on/off.
- **Shift+Click**: Toggle this head and all heads in the same layer.
- **Ctrl+Click**: Add this head to a "compare group" (batch ablation).
- **Right‑click**: Open context menu with "Ablate permanently", "Zero weights", "Add noise", "Permute", "Show impact details".

### Impact Visualization: Before/After Comparison

When a head is toggled, the frontend shows a **before/after comparison panel** that appears as a slide‑in drawer from the right side of the Zone 2 canvas.

**Panel contents:**
1. **Token probability bar chart**: Two rows of bars (blue = before, red = after). Each bar represents a token's probability. Tokens that changed significantly are highlighted.
2. **KL divergence gauge**: A radial gauge showing how much the output distribution changed (0 = identical, 1+ = completely different).
3. **Affected tokens table**: A sortable table listing tokens whose probabilities changed by more than 1%, with the delta shown as a percentage.
4. **Head importance score**: Updated in real time as the user ablates, showing where this head ranks in importance compared to others.

### Chain Ablation Mode

The user can select multiple heads (via Shift+Click or a "Select All in Layer" button) and ablate them simultaneously. The frontend sends a single `ablate_chain` message with all selected heads. The backend runs a single forward pass with all ablations applied.

**Compound effect visualization:** The panel shows not just the individual impact of each head, but the **compound effect**—how much worse is the model when all selected heads are ablated compared to when just one is ablated? This reveals redundancy: if ablating heads A and B together has the same effect as ablating just A, then B is redundant given A.

### Export to Compression Pipeline

Every ablation configuration can be exported as a **pruning recipe**—a JSON file that can be fed directly into your model compression pipeline. The export button is prominently placed in the ablation panel.

**Pruning recipe format:**

```json
{
  "model_id": "bert-base-uncased",
  "ablated_heads": [
    {"layer": 3, "head": 7},
    {"layer": 5, "head": 2}
  ],
  "impact_score": 0.87,
  "kl_divergence": 0.12,
  "recommended": true,
  "rationale": "These heads are redundant; ablating them has minimal impact on output quality."
}
```

This recipe can be imported into the compression pipeline, which will physically remove the heads and create a smaller, faster model.

---

## Feature 22: Anthropic‑Style Circuit Tracer — Implementation

### Purpose

The Circuit Tracer reveals *how attention heads communicate with each other*. It shows the **QK circuit** (what a head "looks for" in keys from previous layers) and the **OV circuit** (what a head "writes" to the output that subsequent heads can read). The result is an attribution graph—a network diagram where nodes are features (transcoder‑decomposed directions in activation space) and edges show how information flows between them.

### Backend Implementation: Three‑Stage Pipeline

The circuit tracing backend has three stages, each implemented as a separate Python module:

**Stage 1 — Transcoder Loading.** The model must have pre‑trained cross‑layer transcoders (CLTs) installed. The backend uses the `circuit‑tracer` library (PyPI package `circuit-tracer`), which inherits from TransformerLens's `HookedTransformer`. Supported models include Pythia‑160M, Gemma‑2‑2B, and Llama‑3.2‑1B (the models for which Anthropic released pre‑trained CLTs).

**Stage 2 — Attribution Graph Computation.** For the given prompt, the model runs a forward pass. At each layer, the transcoders decompose the residual stream into features (sparse, interpretable directions). The attribution algorithm then computes, for each pair of non‑zero features (A, B), the direct effect of A on B: how much would B's activation change if A were zeroed out? This is computed via a linear approximation using the Jacobian of the model. The result is a directed graph with weighted edges.

**Stage 3 — Supernode Formation.** Raw features (hundreds or thousands) are too numerous to visualize. The backend groups features into **supernodes** using clustering based on activation similarity and semantic relatedness. Each supernode gets an auto‑generated label (e.g., "rhyming words," "French language," "punctuation"). This labeling is done via an LLM‑based probe: the backend samples sentences that strongly activate each supernode, then asks a small language model to describe what those sentences have in common.

### Frontend Implementation: Interactive Attribution Graph

The attribution graph is rendered using **react‑force‑graph‑2d** (a React wrapper around D3‑force for 2D graphs) and **react‑force‑graph‑3d** (a React‑Three‑Fiber wrapper for 3D graphs). The user can toggle between 2D and 3D modes.

**2D mode (react‑force‑graph‑2d):**
- Nodes are circles colored by layer (Layer 0 = blue, Layer 5 = green, Layer 11 = red).
- Node size is proportional to activation magnitude.
- Edges are curved arrows colored by weight (darker = stronger connection).
- The force simulation runs in real time—nodes settle into equilibrium over ~2 seconds.
- **Pan and zoom** via mouse wheel and drag.
- **Hover**: Highlights all edges connected to this node, dims others, shows a tooltip with the supernode label and top‑activating tokens.
- **Click**: Pins the node, opens a detail panel showing the supernode's features, activation distribution, and a "Explain this supernode" button that queries an LLM.
- **Drag**: Nodes can be repositioned manually.

**3D mode (react‑force‑graph‑3d with r3f‑forcegraph):**
- The same graph data rendered in a Three.js scene using `@react-three/fiber` and `@react-three/drei`.
- Nodes are 3D spheres with emissive materials that glow based on activation magnitude.
- Edges are tubes with varying radius (wider = stronger).
- The camera orbits the graph automatically (slow rotation) until the user interacts.
- **OrbitControls**: Rotate, pan, zoom with mouse.
- **Post‑processing**: Bloom effect (via `@react-three/postprocessing`) makes high‑activation nodes glow.
- VR mode available via `ForceGraphVR` from the same library.

**Graph interaction features:**
- **Path highlighting**: Click two nodes to highlight the shortest path between them. The path is animated with a pulsing glow.
- **Layer filtering**: A slider lets users show/hide nodes from specific layers, revealing cross‑layer information flow.
- **Edge weight threshold**: A slider filters out weak edges, showing only the strongest connections.
- **Export**: The graph can be exported as JSON (for programmatic analysis), PNG (for papers), or interactive HTML (for sharing).

### JSON Schema for Attribution Graph

The backend outputs attribution graphs in Anthropic's standard JSON format, which is directly compatible with Neuronpedia's frontend. This means users can optionally export their graph and view it on Neuronpedia for comparison with published graphs.

```json
{
  "nodes": [
    {
      "id": "supernode_0",
      "label": "French language features",
      "layer": 4,
      "features": ["f/1024", "f/2048"],
      "activation": 0.87,
      "top_tokens": ["Paris", "France", "bonjour"]
    }
  ],
  "edges": [
    {
      "source": "supernode_0",
      "target": "supernode_5",
      "weight": 0.73,
      "type": "qk"
    }
  ]
}
```

---

## Feature 23: Dodrio‑Style Linguistic Annotation Overlay — Implementation

### Purpose

The Dodrio overlay superimposes linguistic structure on top of the attention heatmap, letting researchers answer: *Does this attention head track syntax? Does it follow the dependency parse tree? Does it recognize coreference chains?*

### Backend Implementation: Linguistic Annotation Pipeline

The backend runs two NLP pipelines on the input text:

**Pipeline 1 — spaCy (English, fast).** Using `en_core_web_trf` (the transformer‑based spaCy model for English), the backend extracts:
- **POS tags**: Part‑of‑speech for each token (NOUN, VERB, ADJ, etc.).
- **Dependency relations**: For each token, its syntactic head and the relation label (nsubj, dobj, amod, etc.).
- **NER tags**: Named entities (PERSON, ORG, GPE, etc.).
- **Noun phrases**: Base noun phrase spans.

**Pipeline 2 — Stanza (multilingual, accurate).** For non‑English text or when higher accuracy is needed, the backend uses Stanza (StanfordNLP) via the `spacy_stanza` wrapper. Stanza provides the same annotations but with state‑of‑the‑art accuracy for 68 languages.

**Coreference resolution** is provided by spaCy's `experimental.coref` pipeline (for English) or the `fastcoref` library. The backend identifies which tokens refer to the same entity and groups them into coreference chains.

**Annotation caching:** Linguistic annotations are cached per text string. If the user changes the text, the annotations are recomputed. If the user changes only the model or layer, the cached annotations are reused.

### Frontend Implementation: Overlay Rendering

The linguistic overlay is rendered on top of the existing attention heatmap (Head View) and as an additional view mode. The overlay uses **PixiJS** (via `@pixi/react`) for high‑performance rendering, as D3 SVG becomes slow with more than 30 tokens.

**Dependency parse overlay:**
- Each dependency relation is drawn as a **curved arrow** from the head token to the dependent token.
- The arrow color indicates the relation type (e.g., `nsubj` = blue, `dobj` = orange, `amod` = green).
- The arrow is drawn **above** the attention heatmap, with 60% opacity so the attention pattern remains visible underneath.
- Hovering an arrow highlights it and shows the relation label in a tooltip.

**POS tag coloring:**
- The token labels along the axes of the heatmap are colored by POS: NOUN = blue, VERB = red, ADJ = green, ADP = gray, etc.
- A legend in the corner shows the color mapping.

**Coreference chain highlighting:**
- When the user toggles "Coreference" mode, all tokens belonging to the same coreference chain are highlighted with matching background colors.
- Connecting lines (dotted) are drawn between coreferring tokens across the sequence.
- Attention weights between coreferring tokens are emphasized with a glow effect.

**Alignment metric:** The overlay computes an **alignment score** for each attention head—how well does the attention pattern correlate with the dependency structure? For a syntactic head, attention should be high between syntactically connected tokens. The alignment score is displayed as a badge (e.g., "Syntax Alignment: 0.87").

**Toggle modes (multi‑select checkboxes):**
- [✓] Show dependency parse
- [✓] Show POS tags
- [ ] Show coreference chains
- [ ] Show named entities
- [ ] Show noun phrase spans
- [✓] Auto‑detect head type (syntactic / semantic / delimiter)

When "Auto‑detect head type" is enabled, the system uses the computed alignment scores to classify each head and displays a badge: "This head appears to be a **syntactic head** (dependency alignment: 0.92)."

---

## Feature 24: Causal Tracing — Knockout Experiment — Implementation

### Purpose

Causal Tracing lets researchers answer: *"Which specific activations caused this token to be predicted? If I change the activation at layer 5, position 3, does the output at position 7 change?"* This is the most rigorous interpretability tool—it doesn't just show correlations (like attention weights), it tests **causation** through systematic intervention.

### Backend Implementation: The Causal Tracing Engine

The causal tracing backend uses **pyvene** (Stanford NLP) for its declarative intervention framework. The core algorithm follows the ROME (Rank One Model Editing) causal tracing procedure, adapted for real‑time interactive use.

**Step‑by‑step algorithm:**

1. **Baseline (clean) run**: The model is run on the input text. All activations are collected and cached.
2. **Corrupted run**: The model is run on a *corrupted* version of the input, where the target token is replaced with noise (e.g., Gaussian noise added to the embedding of a specific token). The corrupted output is collected.
3. **Patching runs**: For each (layer, token_position) pair, the model is run with the *corrupted* input, but the *clean* activation at (layer, token_position) is restored. If restoring this one activation recovers the clean output, then that activation is **causally important** for the prediction.
4. **Causal graph construction**: The patching results are aggregated into a graph showing which (layer, position) pairs causally influence which other (layer, position) pairs. The graph is thresholded—only edges with causal influence above a significance threshold are shown.

**pyvene implementation:**

The backend uses pyvene's `IntervenableConfig` to declaratively specify the patching experiment:

```python
config = IntervenableConfig(
    intervenable_type="ActivationPatching",
    source_representation="block_output",
    intervention_type="SwapIntervention",
    intervention_locations=[
        {"layer": L, "component": "attention_output"} for L in range(num_layers)
    ]
)
```

Then, pyvene's `intervene()` function runs all patching configurations efficiently, parallelizing across layers where possible.

**Optimization for interactive use:** Running patching for every (layer, token) pair is computationally expensive (O(L × N²) for N tokens). The backend uses two optimizations:

1. **User‑directed search**: The user selects a source token and target token. The backend only traces causal paths between these two, reducing the search space from O(L × N²) to O(L).
2. **Gradient‑based approximation**: For a first‑pass estimate (shown immediately), the backend uses integrated gradients to compute an attribution score for each (layer, position) pair in a single backward pass. This is approximate but fast (< 1s). The full patching results are computed in the background and replace the approximation when ready.

### Frontend Implementation: The Causal Graph Explorer

The causal tracing results are displayed as an **interactive causal graph** overlaid on a simplified model diagram.

**Causal graph visualization (D3‑force):**
- The X‑axis represents **token positions** (0 to N‑1).
- The Y‑axis represents **model layers** (0 to L‑1).
- Each (layer, token_position) that had a significant causal effect is shown as a **node** (colored circle).
- Edges are drawn from cause to effect with thickness proportional to causal strength.
- The user's selected source and target tokens are highlighted with gold borders.

**Critical path highlighting:**
The backend identifies the **critical path**—the sequence of (layer, token) pairs that carries the most causal influence from source to target. This path is highlighted with a glowing animation (like a neural pulse traveling along the path). The user can scrub through the path with a slider, seeing how information propagates through the model step by step.

**Node inspection:**
- **Hover** a node: Shows the exact causal influence score and the patching result (e.g., "Restoring this activation recovers 87% of the clean output").
- **Click** a node: Pins it and shows a detail panel with the activation vector visualized as a bar chart.
- **Double‑click** a node: Opens the attention pattern for that layer, filtered to show only the tokens that causally influence this position.

**Knockout experiment mode (game‑like):**
For educational and exploratory use, the tool includes a **"Knockout Challenge"** mode. The system presents a prediction task (e.g., "The model predicts 'Paris' at position 7. Find the activation that, when knocked out, changes this prediction."). The user clicks on nodes to "knock them out" (set activation to zero) and sees if the prediction changes. Points are awarded for finding the minimal set of knockouts that change the output. This gamifies causal tracing and teaches researchers how to think causally about model internals.

**Export to compression:**
The causal graph can be exported as a **pruning guide**: edges with low causal influence indicate redundant connections that can be pruned. Nodes (heads) with no outgoing causal edges are candidates for ablation. This export feeds directly into the compression pipeline.

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|---|---|---|
| **Model loading** | TransformerLens, nnsight | Load and hook GPT‑2/Pythia/Gemma/Llama models |
| **Intervention engine** | pyvene, nnsight edit() | Declarative intervention specification |
| **Attribution graph** | circuit‑tracer (Anthropic) | Compute transcoder‑based attribution graphs |
| **Linguistic pipeline** | spaCy + Stanza | POS, dependency parse, NER, coreference |
| **2D graph visualization** | react‑force‑graph‑2d, PixiJS | Attribution graph, causal graph |
| **3D graph visualization** | react‑force‑graph‑3d, r3f‑forcegraph | 3D circuit view |
| **Attention heatmap** | D3.js, Canvas API | High‑performance attention matrix rendering |
| **Communication** | WebSocket (JSON protocol) | Real‑time intervention streaming |
| **State management** | Zustand | Frontend intervention state |
| **GPU computation** | PyTorch CUDA | Fast forward passes and patching |
| **Caching** | IndexedDB (frontend), Redis (backend) | Model weights, annotations, attention patterns |

---

## Implementation Roadmap for Zone 6

| Phase | Duration | Deliverables |
|---|---|---|
| **Phase 1: Backend Core** | Weeks 1‑3 | Model loading (TransformerLens + nnsight), hook system, ablation engine, WebSocket server |
| **Phase 2: Ablation UI** | Weeks 4‑5 | Ablation grid, toggle interaction, before/after comparison panel, export to compression |
| **Phase 3: Circuit Tracer** | Weeks 6‑8 | circuit‑tracer integration, attribution graph computation, 2D/3D force‑graph visualization |
| **Phase 4: Linguistic Overlay** | Weeks 9‑10 | spaCy/Stanza integration, PixiJS overlay rendering, auto‑detect head type |
| **Phase 5: Causal Tracing** | Weeks 11‑13 | pyvene integration, causal graph visualization, knockout challenge mode |
| **Phase 6: Polish** | Weeks 14‑16 | Performance optimization, caching, accessibility, documentation |