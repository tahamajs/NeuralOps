# 🧠 Output Logits Predictor — Complete Implementation Guide

> *"The Decision Engine" — Showing exactly how the model chooses its next token, from raw logits through temperature, sampling, and softmax to the final probability distribution.*

---

## 📐 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              OUTPUT LOGITS PREDICTOR — DATA FLOW                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  RAW     │    │ TEMP     │    │ SAMPLING │    │ FINAL    │       │
│  │  LOGITS  │───▶│ SCALING  │───▶│ (top‑k/  │───▶│ PROB     │       │
│  │  (LN)    │    │ (÷T)     │    │  top‑p)  │    │ DIST     │       │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │
│       │               │               │               │              │
│       ▼               ▼               ▼               ▼              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              LOGIT SPECTRUM VISUALIZATION                 │       │
│  │    (full vocabulary heatmap / top‑k bar chart)           │       │
│  └──────────────────────────────────────────────────────────┘       │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │         TUNED LENS / LOGIT LENS CROSS‑LAYER VIEW          │       │
│  │    (what each layer "thinks" the next token should be)   │       │
│  └──────────────────────────────────────────────────────────┘       │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │         CALIBRATION & ENTROPY ANALYTICS                   │       │
│  │    (reliability diagram · prediction entropy timeline)   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

The architecture follows the Transformer Explainer's proven design: "the probability visualization component displays the probability distribution over the vocabulary tokens predicted by the GPT‑2 model. It shows not only the final probabilities but also the intermediate steps involved in calculating these probabilities, making the prediction process transparent to users."

---

## 2. Zone‑by‑Zone Implementation Detail

### 2.1 CANVAS — Token Probability Distribution (Primary View)

#### 2.1.1 Component Tree

```
TokenProbabilityPanel (container)
├── ProbabilityBarChart (main visualization — Plotly.js Bar chart)
│   ├── TopKTokenBar (individual horizontal bar, D3‑animated)
│   ├── SelectedTokenIndicator (golden glow on sampled token)
│   └── CutoffLine (vertical dashed line for top‑k boundary)
├── CollapsedView (compact: shows only the probability bars)
│   └── ExpandButton (triggers GSAP Flip animation)
└── ExpandedView (detailed: shows raw logits → scaled logits → probabilities)
    ├── RawLogitColumn (numeric display, monospace)
    ├── ScaledLogitColumn (after temperature division)
    ├── SamplingEffectColumn (shows which tokens were filtered)
    └── FinalProbColumn (normalized softmax output)
```

#### 2.1.2 Framework Selection

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Rendering engine** | `Plotly.js` via `react-plotly.js` | Handles bar charts with built‑in hover, zoom, and export. The Transformer Explainer uses a similar approach with `ProbabilityBars` as a sub‑component for rendering probability bars. |
| **Animation** | `GSAP` (GreenSock) with `Flip` plugin | For smooth collapsed ↔ expanded transitions. The original Transformer Explainer uses exactly this: "The expansion and collapse are animated using GSAP and Flip for smooth transitions." |
| **Bar chart interactivity** | `D3.js` scales + React rendering | D3 handles the mathematical computation (scales, data binding); React owns the DOM. This hybrid approach is documented as the preferred pattern for React‑D3 integration. |
| **Top‑k cutoff visualization** | Custom SVG overlay using `react‑svg` | For drawing the cutoff line and highlighting which tokens survive top‑k filtering. |
| **Sampled token indicator** | `framer‑motion` `motion.div` | Golden pulsing glow on the token actually chosen by the sampling algorithm. |

#### 2.1.3 Data Flow Implementation

The component receives data through a custom React hook `useModelOutput`:

```typescript
// Conceptual data flow — not code, but architecture specification

interface LogitDataFlow {
  // Step 1: Raw logits from the model's final linear layer
  rawLogits: Float32Array;          // Length = vocabulary size (50,257 for GPT‑2)
  
  // Step 2: Temperature‑scaled logits (rawLogits / temperature)
  scaledLogits: Float32Array;       // Computed client‑side for responsiveness
  
  // Step 3: After sampling filter applied
  filteredLogits: Float32Array;     // -Inf for filtered tokens
  
  // Step 4: Final softmax probabilities
  probabilities: Float32Array;      // Sums to 1.0
  
  // Metadata
  tokenStrings: string[];           // Decoded token text
  selectedTokenIndex: number;       // Which token was actually sampled
  temperature: number;              // Current temperature value
  samplingMethod: 'greedy' | 'temperature' | 'top_k' | 'top_p' | 'beam';
  topK: number;                     // k value for top‑k sampling
  topP: number;                     // p value for nucleus sampling
}
```

The critical design decision here: **temperature scaling, sampling, and softmax are computed client‑side in a Web Worker** so the UI remains responsive at 60fps even with a 50K‑token vocabulary. The raw logits arrive from the Django backend via the existing REST API or WebSocket. All subsequent transformations happen in the browser.

The Transformer Explainer follows this same pattern: "Model output data is accessed via the modelData store. Temperature and sampling parameters are accessed via their respective stores. Data is processed and formatted for display. Different visualizations are rendered based on the processed data."

#### 2.1.4 Top‑k Visualization Detail

When top‑k sampling is active, the bar chart renders:

- **Blue bars**: tokens that survive the filter (top‑k highest probabilities)
- **Gray bars** (faded, 20% opacity): tokens that were filtered out
- **Red dashed vertical line**: the exact cutoff position (at the k‑th token)
- **Golden pulsing bar**: the token actually sampled (if in "sample" mode)

The cutoff line position is computed as: `cutoffX = (k / visibleTokenCount) * chartWidth`. The Transformer Explainer implementation: "When top‑k sampling is selected, only the k most likely tokens are kept, and the rest are filtered out (set to -infinity). This is shown by highlighting the top‑k tokens and indicating where the cutoff occurs."

#### 2.1.5 Top‑p (Nucleus) Visualization Detail

For top‑p sampling, the visualization is more complex:

- A **cumulative probability curve** is plotted on a secondary axis
- A **green shaded region** shows the tokens whose cumulative probability ≤ p
- The **p‑threshold line** intersects the cumulative curve at the cutoff point
- Tokens beyond the cutoff are grayed out

The Transformer Explainer: "When top‑p sampling is selected, tokens are kept until their cumulative probability exceeds the threshold p. The component shows token probabilities before filtering, after filtering, and the final normalized probabilities after top‑p is applied."

---

### 2.2 CANVAS 2 — Logit Spectrum (Full Vocabulary Heatmap)

#### 2.2.1 Visualization Strategy

Since the vocabulary is large (50K+ tokens), direct rendering of all tokens is infeasible. Two complementary views are provided:

**View A: Top‑k Bar Chart with Scroll/Zoom**
- Shows the top 50-100 tokens ranked by probability
- Horizontal bars with D3‑based zoom/pan (like a genome browser)
- A "mini‑map" at the bottom shows the full distribution compressed

**View B: Logit Heatmap Grid**
- Tokens are arranged in a 2D grid (like a spectrogram)
- X axis: token index; Y axis: nothing (single row) — or for multi‑token comparison, each row is a different model/layer/timestep
- Color intensity = logit value (blue = negative, white = zero, red = positive)
- Hover reveals the token text and exact logit value
- Uses `Plotly.js` heatmap trace with `colorscale: 'RdBu'`

#### 2.2.2 Performance Architecture

For the heatmap with 50K tokens, a canvas‑based renderer is used instead of SVG. The library of choice is **Deck.gl** (Uber's WebGL‑powered visualization framework) which can render millions of data points at 60fps. Alternatively, **Observable Plot** can be used for a simpler but still performant implementation.

The data flows from the Django backend as a `Float32Array` over WebSocket for real‑time updates, or as a JSON array for REST API calls.

#### 2.2.3 Comparison Mode

When comparing two models (e.g., pre‑compression vs post‑compression), the logit spectrum shows:

- **Row 1**: Model A logits
- **Row 2**: Model B logits
- **Row 3**: Difference (A − B) as a diverging heatmap

This leverages the same infrastructure as the attention pattern comparison features already designed.

---

### 2.3 MATH PANEL — Softmax Formula with Interactive Temperature

#### 2.3.1 Component Architecture

```
SoftmaxMathPanel
├── FormulaRenderer (KaTeX — renders the softmax equation)
│   ├── HighlightedVariable (each variable linked to visualization)
│   └── LiveValueSubstitution (shows actual numbers for current token)
├── TemperatureSlider (shadcn/ui Slider with gradient track)
│   ├── GradientTrack (blue‑to‑red gradient representing cold‑to‑hot)
│   ├── PresetButtons ([0.0] [0.3] [0.7] [1.0] [1.5] [2.0])
│   └── LiveDistributionPreview (mini bar chart updating in real‑time)
├── StepByStepAnimator (GSAP Timeline)
│   ├── Step1: "Raw Logits" → shows the vector before processing
│   ├── Step2: "Divide by Temperature" → shows scaled values
│   ├── Step3: "Apply Sampling Filter" → shows which tokens survive
│   ├── Step4: "Softmax" → shows final probabilities
│   └── Step5: "Sample Token" → shows the selected output
└── SamplingMethodSelector (radio buttons or segmented control)
    ├── Greedy (argmax)
    ├── Temperature (sample from distribution)
    ├── Top‑k (with k slider)
    ├── Top‑p (with p slider)
    ├── Beam Search (with beam width slider + tree visualization)
    └── Repetition Penalty (slider, 1.0–2.0)
```

#### 2.3.2 KaTeX Formula Rendering

The formula is rendered using **KaTeX** (faster than MathJax, better for interactive applications):

```
P(token_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
```

Each variable in the formula is wrapped in a `<span>` that highlights (glows) when the user hovers over the corresponding UI element. For example, hovering over the temperature slider highlights `T` in the formula; hovering over a bar in the probability chart highlights `P(token_i)`.

#### 2.3.3 Temperature Slider Design

The slider uses **shadcn/ui Slider** with a custom gradient track:

- Track gradient: `linear-gradient(to right, #3b82f6, #ef4444)` (blue → red)
- Blue end = low temperature (deterministic, peaked distribution)
- Red end = high temperature (random, uniform distribution)
- Thumb shows the current temperature value
- Preset buttons for quick switching

A **live mini‑preview** (a tiny horizontal bar chart, approximately 100px wide) updates in real‑time as the user drags the slider, showing how the distribution shape changes. This is implemented using a lightweight D3‑based `canvas` element for 60fps updates.

The approach is directly inspired by the Transformer Explainer: "The raw logits are divided by a temperature parameter, which controls the randomness of predictions: Lower temperature (< 1.0) makes the distribution more peaked (more deterministic). Higher temperature (> 1.0) makes the distribution more uniform (more random)."

#### 2.3.4 Step‑by‑Step Animator Implementation

Each step is a GSAP `Timeline` animation. The user can scrub through the timeline with a slider (like video playback). Implementation details:

| Step | Animation | Duration | Trigger |
|------|-----------|----------|---------|
| 1. Raw Logits | Numbers appear in monospace grid | 0.3s | Auto‑play or click |
| 2. Temperature | Each number divides by T, color shifts | 0.3s | Auto‑advance |
| 3. Sampling | Filtered tokens fade to gray | 0.3s | Auto‑advance |
| 4. Softmax | Numbers normalize, bars grow | 0.5s | Auto‑advance |
| 5. Sample | Golden glow on selected token | 0.3s | Auto‑advance |

The GSAP `Timeline` is controlled by a `<input type="range">` scrubber and play/pause buttons.

---

### 2.4 CONTROLS — Sampling Method Selector

#### 2.4.1 Sampling Method Architecture

```
SamplingControls
├── MethodSelector (segmented button group)
│   ├── Greedy: "Always pick the highest probability token"
│   ├── Temperature: "Sample from the full distribution"
│   ├── Top‑K: "Sample from top K tokens only"
│   ├── Top‑P: "Sample from tokens until cumulative probability > P"
│   └── Beam Search: "Maintain multiple hypotheses"
├── ParameterSliders (conditionally rendered)
│   ├── Temperature: [0.0 ––––––––●–––––––– 2.0]
│   ├── Top‑K: [1 –––––●––––––––––––––– 100]
│   ├── Top‑P: [0.0 ––––––––●–––––––– 1.0]
│   └── Beam Width: [1 ––●–––––––––––––––– 10]
├── RepetitionPenaltySlider: [1.0 –––––●–––– 2.0]
│   └── Explanation tooltip: "Penalizes tokens that have already appeared"
├── LiveEffectPreview (small bar chart showing current distribution)
└── ResetButton (return to default settings)
```

#### 2.4.2 Beam Search Tree Visualization

When beam search is selected, an additional component renders:

```
BeamSearchTree (D3.js tree layout)
├── TreeNode (represents a partial hypothesis)
│   ├── TokenText (the token at this node)
│   ├── ScoreIndicator (cumulative log‑probability)
│   ├── ActiveIndicator (golden border if this path survived)
│   └── PrunedIndicator (red X if this path was eliminated)
├── TreeEdges (lines connecting parent to children, thickness ∝ score)
└── FinalHypotheses (the top‑k completed sequences at the end)
```

This component uses **D3.js tree layout** (`d3.tree()`) with React rendering. Each node in the tree represents a partial hypothesis. Nodes that survive beam pruning have gold borders; pruned nodes have red X marks. The final hypotheses are highlighted at the bottom.

The beam search visualization "creates interactive tree diagrams showing how the model explores different sequence hypotheses during decoding. Each node represents a token in the sequence. Nodes show the token and its score." The implementation "converts beam search decoding results into an interactive tree structure rendered using D3.js."

#### 2.4.3 Repetition Penalty Implementation

The repetition penalty slider applies a penalty to logits of tokens that have already appeared in the generated sequence. The formula is:

```
logit_i = logit_i - penalty · 1[token_i already appeared]
```

The visualization shows:
- Previously generated tokens highlighted in a different color
- The penalty amount as a red overlay on those tokens' bars
- A "penalty severity" indicator (how much the probability of repeated tokens drops)

The repetition penalty mechanism "penalizes previously chosen tokens to minimize repetition."

---

### 2.5 ANALYTICS — Prediction Entropy Over Time

#### 2.5.1 Entropy Timeline Chart

For autoregressive generation, the system tracks prediction entropy at each step:

```
H_t = − Σ_i P(token_i) · log₂ P(token_i)
```

This is visualized as a **line chart** using Plotly.js:

- **X axis**: generation step (token position)
- **Y axis**: entropy (bits)
- **Blue line**: prediction entropy
- **Red dashed line**: maximum possible entropy (log₂|V|)
- **Green shaded region**: "confident" zone (entropy < 1 bit)
- **Yellow shaded region**: "uncertain" zone (entropy > 5 bits)
- **Orange dots**: steps where the model changed its mind (top prediction flipped)

Below the chart, a **token ribbon** shows the actual generated tokens at each step, color‑coded by the model's confidence.

This approach is grounded in information theory: "autoregressive generation is essentially a process of progressively reducing sequential uncertainty through successive observations. Each token generation represents a new observation. As a direct measure of uncertainty, entropy effectively reflects the change in predictive uncertainty throughout the process."

#### 2.5.2 Architecture for Real‑Time Tracking

During generation, the system maintains a **ring buffer** of the last 100 generation steps:

1. Each generation step pushes `{step, tokens, probabilities, entropy}` into the buffer
2. The Plotly chart subscribes to buffer updates via React context
3. Plotly's `extendTraces` method is used for efficient incremental updates without full re‑render
4. The buffer persists across generation sessions for comparison

---

### 2.6 ANALYTICS — Calibration Analysis (Reliability Diagram)

#### 2.6.1 What a Reliability Diagram Shows

A reliability diagram "plots the observed frequency of an event against the predicted probability." If the model is perfectly calibrated, the plot follows the diagonal (identity line). Deviations indicate over‑confidence (below the diagonal) or under‑confidence (above the diagonal).

#### 2.6.2 Implementation Architecture

```
ReliabilityDiagram
├── PlotArea (Plotly.js scatter plot)
│   ├── IdentityLine (black dashed diagonal, y = x)
│   ├── CalibrationCurve (blue line with error bars)
│   ├── HistogramMarginals (bar chart showing # samples per bin)
│   └── PerfectCalibrationRegion (green shaded band around identity)
├── MetricsPanel
│   ├── ECE (Expected Calibration Error)
│   ├── MCE (Maximum Calibration Error)
│   └── Brier Score
└── LayerSelector (dropdown to choose which layer to evaluate)
```

#### 2.6.3 Data Flow

The calibration data flows from the Django backend:

1. **Backend**: During training, the model outputs predicted probabilities and actual outcomes (correct/incorrect) for validation data. These are stored in the `TrainingSnapshot` model or a dedicated `CalibrationData` table.

2. **API**: `GET /api/experiments/{run_id}/calibration/?layer=5` returns:
   ```json
   {
     "layer": 5,
     "bins": [
       {"confidence_bin": [0.0, 0.1], "accuracy": 0.05, "count": 120},
       {"confidence_bin": [0.1, 0.2], "accuracy": 0.18, "count": 200},
       ...
     ],
     "ece": 0.034,
     "mce": 0.12,
     "brier_score": 0.089
   }
   ```

3. **Frontend**: The Plotly scatter plot is constructed from the bin data. The x‑axis represents confidence (predicted probability), the y‑axis represents accuracy (observed frequency). A perfectly calibrated model follows y = x.

4. **Interactive**: Users can select different layers to see how calibration evolves — earlier layers tend to be less calibrated, later layers (after more processing) become better calibrated.

#### 2.6.4 Visual Design Details

- The calibration curve is rendered with **error bars** showing ±1 standard deviation within each bin
- A **histogram** in the margins shows the number of samples per confidence bin
- The **ECE** (Expected Calibration Error) is prominently displayed as a numeric badge
- Hovering over a point shows: confidence range, actual accuracy, sample count
- A "Compare Mode" allows overlaying two models' calibration curves on the same plot

---

### 2.7 INTERPRETABILITY — Tuned Lens Integration

#### 2.7.1 What Tuned Lens Does

The tuned lens "allows us to peek at the iterative computations a transformer uses to compute the next token." Unlike the logit lens (which applies the final unembedding layer directly to intermediate representations), the tuned lens trains a small affine transformation for each layer that translates that layer's hidden state into the vocabulary space, correcting for representational drift.

#### 2.7.2 Architecture

```
TunedLensPanel
├── LayerSelector (vertical list of all layers)
│   └── Each layer row shows a miniature word cloud of top‑k predictions
├── MainView (shows predictions for selected layer)
│   ├── TokenPredictionBars (top‑k tokens with probability bars)
│   ├── ComparisonWithFinalLayer (overlay showing how predictions differ)
│   └── LayerTransitionAnimation (scrubber showing prediction evolution)
├── TunedLensMetrics
│   ├── Perplexity (how good are predictions at this layer?)
│   ├── KL Divergence (vs final layer distribution)
│   └── Top‑1 Accuracy (does it predict the same token as final?)
└── TrainTunedLensButton (staff only — triggers training job)
```

#### 2.7.3 Data Flow Implementation

The tuned lens data is pre‑computed on the backend:

1. **Training**: The tuned lens affine probes are trained offline using the `AlignmentResearch/tuned-lens` library. "This package provides a simple interface for training and evaluating tuned lenses." The training produces a small affine transformation `(W_layer, b_layer)` for each layer.

2. **Storage**: The trained probes are stored in the `model_versioning` system as artifacts associated with a specific model version.

3. **API**: `POST /api/experiments/tuned-lens/` with body `{model_version_id, text, layers: [0,1,2,...]}` returns for each layer the top‑k predicted tokens and their probabilities.

4. **Frontend**: The `TunedLensPanel` component renders each layer as a miniature bar chart. Clicking a layer expands it to show the full prediction distribution.

#### 2.7.4 Visual Design

The tuned lens view is organized as a **vertical timeline**:

- **Left column**: layer index (0, 1, 2, ..., 12)
- **Main column**: miniature horizontal bar chart of top‑10 predicted tokens for that layer
- **Color coding**: the token that matches the **final prediction** is highlighted in green; tokens that are "distractions" (high probability early but later discarded) are shown in gray
- **Connection lines**: D3 curves connect the same token across layers, showing how its probability evolves (like the Semantic Pathway trajectories)

The tuned lens is closely related to the logit lens: "the logit lens method allows exploration of the residual stream in transformer networks by treating middle layer hidden states as the final layer output." The tuned lens improves on this by correcting for representational drift.

---

## 3. Backend Integration

### 3.1 Django Models

```python
# gpu_benchmark/models.py — add these models

class LogitSnapshot(models.Model):
    """A single logit vector captured at a specific point."""
    run = models.ForeignKey('experiment.TrainingRun', on_delete=models.CASCADE)
    layer = models.IntegerField()
    head = models.IntegerField(null=True)
    token_position = models.IntegerField()
    logits = models.BinaryField()  # Serialized Float32Array (50K floats)
    temperature = models.FloatField(default=1.0)
    created_at = models.DateTimeField(auto_now_add=True)


class TunedLensProbe(models.Model):
    """Trained affine probe for a specific layer of a model."""
    model_version = models.ForeignKey('model_versioning.ModelVersion', on_delete=models.CASCADE)
    layer = models.IntegerField()
    weight = models.BinaryField()  # Serialized weight matrix
    bias = models.BinaryField()    # Serialized bias vector
    trained_on_dataset = models.CharField(max_length=200)
    perplexity = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


class CalibrationData(models.Model):
    """Pre‑computed calibration bins for reliability diagrams."""
    run = models.ForeignKey('experiment.TrainingRun', on_delete=models.CASCADE)
    layer = models.IntegerField()
    bins = models.JSONField()  # [{confidence_range, accuracy, count}]
    ece = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
```

### 3.2 REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/logits/predict/` | Run model inference and return logits |
| `GET` | `/api/logits/history/{run_id}/` | Get logit snapshots for a training run |
| `GET` | `/api/tuned-lens/probes/{model_version_id}/` | Get available tuned lens probes |
| `POST` | `/api/tuned-lens/predict/` | Get tuned lens predictions for input text |
| `GET` | `/api/calibration/{run_id}/` | Get calibration data for reliability diagram |
| `POST` | `/api/logits/compare/` | Compare logits between two models/layers |

### 3.3 WebSocket Endpoints

| Channel | Event | Payload |
|---------|-------|---------|
| `ws/logits/{run_id}/` | `logit_step` | `{step, logits, probabilities, entropy, token}` |
| `ws/logits/{run_id}/` | `generation_complete` | `{tokens, entropy_timeline, calibration}` |

---

## 4. Frontend Component Implementation Details

### 4.1 Key React Hooks

| Hook | Purpose | Returns |
|------|---------|---------|
| `useModelOutput(runId)` | Fetch/stream logits from backend | `{rawLogits, probabilities, loading, error}` |
| `useTemperature(defaultT)` | Manage temperature state with debounce | `{temperature, setTemperature, debouncedTemperature}` |
| `useSamplingMethod()` | Manage sampling method and parameters | `{method, params, setMethod, setParam}` |
| `useEntropyTracker()` | Ring buffer for entropy timeline | `{entropyData, addStep, reset}` |
| `useTunedLens(modelVersionId, text)` | Fetch tuned lens predictions | `{layerPredictions, loading}` |
| `useCalibrationData(runId)` | Fetch calibration bins | `{calibrationData, ece, loading}` |

### 4.2 Web Worker for Client‑Side Computation

The softmax, temperature scaling, and sampling computations are performed in a **Web Worker** to prevent main‑thread blocking:

```
// Architecture specification for the LogitWorker

The LogitWorker:
1. Receives {rawLogits: Float32Array, temperature: number, method: string, params: object}
2. Performs temperature scaling: scaled = rawLogits / temperature
3. Applies sampling filter (top‑k sets non‑top‑k to -Infinity; top‑p sets to -Infinity after cumulative threshold)
4. Computes softmax with numerical stability (subtracts max before exp)
5. Returns {probabilities: Float32Array, filteredIndices: number[], selectedIndex: number}
6. Posts result back to main thread via postMessage()
```

This architecture ensures the UI never freezes, even when processing the full 50K vocabulary.

### 4.3 GSAP Animation Architecture

For the collapsed ↔ expanded transition of the probability panel:

```
1. User clicks expand button
2. GSAP Flip.getState() captures current positions of all probability bars
3. Container div grows to expanded size
4. New elements (raw logit column, scaled logit column, sampling effect column) fade in
5. GSAP Flip.from() animates the bars from collapsed positions to expanded positions
6. Total animation duration: 0.4s with ease: "power2.inOut"
```

This is the exact approach used by Transformer Explainer: "The expansion and collapse are animated using GSAP and Flip for smooth transitions."

### 4.4 Color System

| Element | Color | Meaning |
|---------|-------|---------|
| Top‑k surviving tokens | `#3b82f6` (blue‑500) | Active, considered for sampling |
| Filtered tokens | `#64748b` (slate‑500) at 20% opacity | Eliminated by sampling |
| Sampled token | `#f59e0b` (amber‑500) with pulsing glow | The chosen next token |
| Cutoff line | `#ef4444` (red‑500) dashed | Top‑k boundary |
| Cumulative curve | `#22c55e` (green‑500) | For top‑p visualization |
| High entropy zone | `#f59e0b` (amber‑200) background | Uncertainty |
| Low entropy zone | `#22c55e` (green‑200) background | Confidence |
| Calibration identity line | `#000000` (black) dashed | Perfect calibration |

---

## 5. Integration with Other Platform Components

| Platform Component | Integration Point |
|-------------------|-------------------|
| **experiment** | Logit snapshots are saved as part of training runs; calibration data is computed during evaluation |
| **model_versioning** | Tuned lens probes are stored as model artifacts associated with a specific version |
| **leaderboards** | Calibration metrics (ECE, Brier score) are included in benchmark scoring |
| **notes** | One‑click "Save Analysis" creates a ResearcherNote with embedded probability charts |
| **paper** | "Export to Paper" generates LaTeX figures of reliability diagrams and entropy timelines |
| **community** | "Share this Prediction" creates a post in the attention pattern feed |
| **gpu_marketplace** | Logit computation can be offloaded to rented GPUs for large‑vocabulary models |

---

## 6. Performance Considerations

| Concern | Solution |
|---------|----------|
| 50K vocabulary rendering | Use canvas‑based renderer (Deck.gl) instead of SVG; only render top‑50 visible tokens for bar chart |
| Real‑time temperature slider | Web Worker computes softmax; debounce slider input at 16ms (60fps) |
| Multiple open visualizations | Use React.lazy + Suspense for code splitting; each panel is independently lazy‑loaded |
| Large logit history | Ring buffer capped at 100 steps; older data compressed and sent to backend |
| Cross‑tab synchronization | BroadcastChannel API for syncing temperature/sampling state across open tabs |

---

## 📊 Summary: Complete Feature Map

| # | Feature | Zone | Primary Technology |
|---|---------|------|-------------------|
| 1 | Token Probability Bar Chart | Canvas | Plotly.js + GSAP Flip animation |
| 2 | Raw Logit → Scaled → Filtered → Final columns | Canvas (Expanded) | Custom React components + GSAP |
| 3 | Top‑k cutoff line + token highlighting | Canvas | SVG overlay + framer‑motion |
| 4 | Top‑p cumulative curve + threshold | Canvas | Plotly.js secondary axis |
| 5 | Logit Spectrum Heatmap | Canvas 2 | Deck.gl (WebGL) or Plotly.js |
| 6 | Comparison mode (A − B diverging heatmap) | Canvas 2 | Plotly.js subplots |
| 7 | KaTeX formula renderer with live highlighting | Math Panel | KaTeX + React hover events |
| 8 | Temperature slider with gradient track + presets | Math Panel | shadcn/ui Slider + Tailwind gradient |
| 9 | Step‑by‑step animation (GSAP Timeline) | Math Panel | GSAP Timeline + scrubber |
| 10 | Sampling method selector (greedy/temp/top‑k/top‑p/beam) | Controls | shadcn/ui ToggleGroup |
| 11 | Beam search tree visualization | Controls (conditional) | D3.js tree layout |
| 12 | Repetition penalty slider | Controls | shadcn/ui Slider |
| 13 | Live mini‑preview of distribution | Controls | D3 canvas (60fps) |
| 14 | Prediction entropy timeline chart | Analytics | Plotly.js line chart + ring buffer |
| 15 | Token ribbon (generated tokens with confidence color) | Analytics | Custom React component |
| 16 | Reliability diagram (calibration curve) | Analytics | Plotly.js scatter + histogram |
| 17 | ECE / MCE / Brier Score display | Analytics | shadcn/ui Card + Badge |
| 18 | Tuned Lens layer‑by‑layer prediction view | Interpretability | Custom React + D3.js connections |
| 19 | Tuned Lens comparison with final layer | Interpretability | OverlayPlot component |
| 20 | Web Worker for client‑side softmax computation | Infrastructure | Web Worker API + Float32Array |