# 🎛️ Zone 4: Control Panel — Complete Implementation Specification

## Overview

Zone 4 is the **orchestration hub** of the entire Attention Dashboard. It sits as a vertical panel (typically 320–380px wide) on the left side of the main content area, between the global navigation bar (Zone 1) and the visualization canvas (Zone 2). This zone is the **primary interaction surface** where researchers configure what they see, launch experiments, and export their findings. Unlike Zone 1 which is "read‑oriented" (showing what's currently active), Zone 4 is "write‑oriented" — every interaction here changes the state of the entire dashboard.

The panel follows the **"Control Room" metaphor**: just as a mission control center has sliders, buttons, and panels that affect the main display, Zone 4 provides the tactile controls that shape the attention visualization. Research on complex dashboard design emphasizes that "providing value" requires "different logics, such as investigation, inquiry, finding, discovery, and persuasion" — Zone 4 enables all of these modes through its three core features.

---

## 📐 Architecture & Technology Stack

### 2.1 Core Technology Decisions

The Control Panel is built on a carefully selected stack of modern frameworks chosen for performance, developer experience, and the unique demands of real‑time ML visualization:

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **UI Framework** | React 18 with Concurrent Features | `useTransition` for non‑blocking state updates when changing layers/heads; `useDeferredValue` for debouncing text input without blocking the UI thread; Suspense boundaries for progressive loading of heavy visualization components |
| **State Management** | Zustand 5 + Immer middleware | Zustand's minimal API avoids the boilerplate of Redux while providing "speed and production reliability" for dashboards with "dozens of interactive components". Immer enables mutable‑style updates to deeply nested state (the attention context has 15+ nested fields) without spreading nightmares |
| **Animation** | Framer Motion 11 | 60fps animations for slider transitions, button feedback, and panel expand/collapse; `layoutId` for shared layout animations when the layer/head changes; `AnimatePresence` for exit animations on thumbnail previews |
| **Computation Offload** | Comlink + Web Workers | Attention matrix computation (Q×K^T, softmax, ×V) runs in a Web Worker via Comlink's transparent RPC — "offloading heavy computations to background threads while keeping your UI silky smooth" |
| **Model Inference** | Transformers.js (WASM/WebGPU backend) | Runs HuggingFace models directly in the browser via WebAssembly or WebGPU, with "zero dependency" and "low latency inference". Models cached in IndexedDB for instant reload |
| **Model Caching** | IndexedDB via Transformers.js cache API | "IndexedDB is a real attempt to give the browser a database" — model weights (up to 2GB+) are persisted locally so subsequent loads are near‑instantaneous |
| **Virtual Scrolling** | @tanstack/react-virtual | The model selector dropdown may contain hundreds of HuggingFace models — virtual rendering ensures smooth scrolling at 60fps |
| **Search** | Orama (hybrid full‑text + vector search) | For the model selector's search functionality, Orama provides "full-text, vector, and hybrid search with a unique API" in less than 2KB — runs entirely in the browser |
| **Data Persistence** | PGlite (PostgreSQL in WASM) for local‑first sync | Enables SQL queries on the client for exploration history, saved configurations, and offline support — "run Postgres in the browser, Node.js, Bun and Deno, with no need to install any other dependencies" |

### 2.2 State Architecture

The entire Zone 4 shares a single Zustand store (`useAttentionContext`) with the rest of the dashboard. The store is organized into slices for modularity:

```
useAttentionContext
├── modelSlice: { modelId, modelSource, modelVersion, modelCache }
├── attentionSlice: { attentionType, attentionConfig, supportedTypes }
├── textSlice: { inputText, tokens, tokenizer }
├── navigationSlice: { selectedLayer, selectedHead, activeView }
├── comparisonSlice: { compareModelId, compareAttentionType }
├── historySlice: { pastEntries, futureEntries, starredEntries }
└── uiSlice: { sliderPreviewVisible, panelCollapsed, activeTab }
```

Each slice uses Immer for immutable updates. Changes to any slice automatically synchronize to the URL via a custom middleware, and to the backend (for authenticated users) via a debounced sync hook.

---

## 🎚️ Feature 13: Unified Layer/Head/Input Selector

### 3.1 Purpose & Design Philosophy

The Unified Selector is the most frequently used control in the dashboard. Researchers rapidly iterate through layers and heads to find interesting patterns, then type custom text to test hypotheses. The design follows three principles: **speed** (changes should feel instantaneous), **preview** (users should see what they're selecting before committing), and **precision** (both coarse and fine‑grained control should be available).

Traditional range sliders are insufficient for this use case because they provide no preview of the selection and require trial‑and‑error clicking. The Unified Selector innovates with **live thumbnail previews** that render miniature attention heatmaps as the user drags — inspired by video editing timeline thumbnails and the "PreviewSlider" pattern used in design tools.

### 3.2 Component Tree

```
UnifiedSelector (container)
├── TabBar: [Layer Selector | Head Selector | Text Input]
├── LayerSelectorPanel
│   ├── LayerThumbnailStrip (horizontal scroll of mini heatmaps)
│   ├── LayerSlider (range input with custom thumb)
│   ├── LayerStepper (‑ / + buttons for fine control)
│   └── LayerQuickJump (numeric input for direct entry)
├── HeadSelectorPanel
│   ├── HeadGrid (grid of colored dots, size = importance)
│   ├── HeadSlider (range input)
│   ├── HeadStepper (‑ / + buttons)
│   └── HeadQuickJump (numeric input)
└── TextInputPanel
    ├── TextArea (expandable input)
    ├── TokenCounter (live character/token count)
    ├── RecentTexts (dropdown of last 10 inputs)
    └── PresetTexts (curated examples: "Simple sentence", "Long document", "Code snippet")
```

### 3.3 Layer Selector — Live Preview Thumbnail Strip

This is the most innovative UI element in Zone 4 and deserves detailed specification.

#### Visual Layout
The Layer Selector displays a **horizontal strip of miniature attention heatmaps** — one thumbnail per layer. The strip is scrollable (via mouse wheel, drag, or touch swipe) and the currently selected layer's thumbnail is highlighted with a glowing border.

```
┌─────────────────────────────────────────────┐
│ LAYER SELECTOR                              │
│                                             │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌═════┐ ┌─────┐   │
│ │ L0  │ │ L1  │ │ L2  │ ║ L3  ║ │ L4  │ → │
│ │ ░░  │ │ ▓▓  │ │ ░▓  │ ║ ▓░  ║ │ ░░  │   │
│ └─────┘ └─────┘ └─────┘ └═════┘ └─────┘   │
│ ◄───────────────────●─────────────────────► │
│ Layer: [ 3 ]  [ ‹ ] [ › ]                  │
└─────────────────────────────────────────────┘
```

#### Thumbnail Generation Pipeline
Each thumbnail is a 48×48px SVG rendered on‑the‑fly. The generation pipeline uses a **dedicated Web Worker** (via Comlink) to avoid blocking the main thread:

1. **Worker receives**: layer index, attention matrix data (from the current model's forward pass), token count
2. **Worker computes**: a downsampled 48×48 matrix using bilinear interpolation
3. **Worker returns**: a Float32Array of 2304 values (48×48) representing normalized attention weights
4. **Main thread renders**: a `<canvas>` element colored with the selected color scale (heat/blues/viridis)

For models with many layers (e.g., 24‑layer GPT‑2), thumbnails are rendered **lazily** — only the visible portion of the strip (plus 2 thumbnails on each side for scroll buffer) is computed. As the user scrolls, new thumbnails stream in via a `requestIdleCallback` scheduler.

#### Performance Budget
- Thumbnail generation: < 16ms per thumbnail (Web Worker time)
- Canvas rendering: < 4ms per thumbnail (main thread)
- Strip scroll: 60fps maintained via CSS `will-change: transform` and GPU‑accelerated scrolling
- Memory: at most 10 thumbnails in memory at once (visible + buffer), ~22KB each = ~220KB total

#### Interaction Design
- **Hover**: thumbnail enlarges to 72×72px via a smooth `scale(1.5)` transform with a 150ms spring animation. A tooltip appears: "Layer 3 · 12 heads · Entropy: 2.34"
- **Click**: selects the layer. The thumbnail gains a `ring-2 ring-neon-blue shadow-lg shadow-neon-blue/30` border. The main visualization (Zone 2) updates via a crossfade transition (200ms).
- **Drag**: the slider thumb can be dragged. As it moves, the corresponding thumbnail pulses briefly to indicate the current position.
- **Keyboard**: Arrow keys (← →) change the layer by 1. Shift+Arrow changes by 5. The slider thumb animates smoothly to the new position.

#### Fallback States
- **Model not yet loaded**: thumbnails show a shimmer skeleton (pulsing gray gradient) — identical to the loading state used in YouTube and Netflix thumbnails
- **Computation in progress**: thumbnails show a progress ring (SVG circle with animated dashoffset) indicating "Computing attention..."
- **Computation failed**: thumbnail shows a warning icon (⚠️) with a "Retry" button on hover

### 3.4 Head Selector — Interactive Head Grid

Rather than a second slider, heads are selected through a **visual grid of colored dots** arranged in a 4×3 grid (for 12‑head models) or a 4×4 grid (for 16‑head models). This design was chosen because researchers often want to compare specific heads (e.g., "Heads 3 and 7 look similar") — a grid enables rapid visual scanning.

#### Visual Design
Each head is represented as a **circle** (24px diameter) with:
- **Fill color**: the head's "specialization type" color (positional = blue, semantic = red, delimiter = green, induction = purple, unknown = gray)
- **Size**: proportional to the head's importance score (ranges from 16px to 32px)
- **Border**: the currently selected head has a 2px white border with a neon‑blue glow
- **Opacity**: heads with very low importance (< 0.1) are rendered at 30% opacity

```
┌──────────────────────────────────────┐
│ HEAD SELECTOR                        │
│                                      │
│   ●  ●  ●  ●     H0  H1  H2  H3     │
│   ○  ●  ○  ●     H4  H5  H6  H7     │
│   ○  ○  ●  ○     H8  H9  H10 H11    │
│                                      │
│ Head: [ 7 ]  [ ‹ ] [ › ]            │
└──────────────────────────────────────┘
```

#### Interaction Design
- **Hover**: the circle enlarges to 32px, and a tooltip appears: "Head 7 · Induction Head · Importance: 0.87 · Entropy: 1.23"
- **Click**: selects the head. The circle animates with a "pop" effect (scale 1.0 → 1.3 → 1.0 over 200ms).
- **Keyboard**: Up/Down arrows navigate through heads in column order; Left/Right in row order.
- **Multi‑select** (Shift+Click): selects multiple heads for comparison mode. Selected heads appear in the comparison panel (Zone 5).

#### Dynamic Head Count
For models with variable numbers of heads (e.g., ALBERT with 64 heads), the grid dynamically adjusts: 8×8 for 64 heads, with the ability to zoom in/out via pinch gesture or scroll wheel.

### 3.5 Text Input Panel

The text input is where researchers test hypotheses: "What happens to attention if I change 'The cat sat on the mat' to 'The dog sat on the mat'?" The panel supports both quick edits and batch processing.

#### Component Design
- **TextArea**: an auto‑resizing `<textarea>` that grows from 2 lines to 12 lines. Styled with the glass aesthetic (`bg-white/5 border border-white/10`). Placeholder text: "Enter text to analyze attention patterns..."
- **Token Counter**: a live counter showing "Characters: 47 | Tokens: 12 | Max: 512". Changes color from green → yellow → red as the limit approaches.
- **Preset Library**: a dropdown of curated texts: "Simple sentence", "Paragraph with coreference", "Code snippet", "Multi‑lingual", "Mathematical notation". Each preset includes a brief description of what attention phenomena it demonstrates.
- **Recent Texts**: the last 10 analyzed texts, stored in `localStorage`. Displayed as a dropdown with preview snippets. Click any to reload.

#### Debouncing Strategy
Text input is debounced at **500ms** using React 18's `useDeferredValue` hook. This means:
- The text in the input updates immediately (responsive typing)
- The attention computation is deferred until the user pauses typing
- The UI never blocks — the user can continue interacting with other controls while computation runs

#### Batch Mode
A "Batch" toggle enables pasting **multiple texts** (one per line). The system processes each text through the current model and generates a **comparison view** showing how attention patterns differ across inputs. This is invaluable for linguistic analysis (e.g., comparing active vs. passive voice, or different languages).

---

## ⚡ Feature 14: "Train with This Attention" Button

### 4.1 Purpose & Design Philosophy

This button is the **bridge between exploration and experimentation**. It transforms the attention dashboard from a passive analysis tool into an active research platform. The philosophy is: "Every insight should be immediately testable." When a researcher discovers an interesting attention pattern, they should be able to launch a training experiment with one click, pre‑configured to reproduce or extend that pattern.

This feature is what connects Zone 4 to the rest of the NeuralOps platform — it's the implementation of the "closed loop" that makes the dashboard not just a visualization tool, but a **research accelerator**.

### 4.2 Visual Design

The button is the most visually prominent element in Zone 4. It uses a **gradient background** (from neon‑blue to purple‑500), a subtle glow animation (`animate-glow-pulse`), and an icon of a rocket (🚀) or lightning bolt (⚡).

```
┌──────────────────────────────────────────┐
│                                          │
│   ⚡  TRAIN WITH THIS ATTENTION           │
│   ───────────────────────────────────     │
│   Model: BERT‑base‑uncased               │
│   Attention: Galerkin (rank=64)          │
│   Text: "The quick brown fox..."         │
│   Layer: 5 · Head: 7                     │
│                                          │
│   [Configure & Launch]  [Quick Launch]   │
│                                          │
└──────────────────────────────────────────┘
```

When expanded (by clicking the main button or the expand arrow), it reveals a compact configuration panel:

```
┌──────────────────────────────────────────┐
│   ⚡  TRAIN WITH THIS ATTENTION    [−]   │
│   ───────────────────────────────────     │
│   Experiment Name:                       │
│   ┌──────────────────────────────────┐   │
│   │ Reproduce L5-H7 on BERT-base     │   │
│   └──────────────────────────────────┘   │
│                                          │
│   Task: [GLUE-MRPC ▾]                   │
│   Epochs: [3]  Batch: [16]  LR: [2e-5] │
│   ───────────────────────────────────     │
│   ☐ Compare with standard MHA           │
│   ☐ Track attention during training     │
│   ☐ Auto‑generate report                │
│   ───────────────────────────────────     │
│   [Save as Draft]    [Launch Experiment] │
└──────────────────────────────────────────┘
```

### 4.3 Pre‑Filled Configuration Pipeline

When the user clicks "Configure & Launch", the system assembles a configuration object by reading from multiple sources:

#### Source 1: Current Attention Context (Zone 1 + Zone 4)
- `model_name`: from the Model chip in Zone 1
- `attention_type`: from the Attention chip in Zone 1
- `attention_config`: rank, order, temperature, etc. from Zone 4 controls
- `sample_text`: from the Text Input in Zone 4
- `selected_layer`: from the Layer Selector
- `selected_head`: from the Head Selector

#### Source 2: User Profile & Preferences
- `default_task`: from the user's profile preferences (stored in PGlite local‑first database)
- `default_epochs`, `default_batch_size`, `default_learning_rate`: from user's training history (most frequently used values)
- `preferred_hardware`: GPU preference from the `gpu_marketplace` usage history

#### Source 3: Platform Intelligence
- `suggested_experiment_name`: auto‑generated from the context (e.g., "Reproduce_L5‑H7_BERT‑base_Galerkin_r64")
- `related_experiments`: fetched from the `experiment` app — other experiments with similar configurations
- `estimated_cost`: computed from the `gpu_marketplace` current pricing and the selected configuration
- `estimated_duration`: computed from historical run data for similar experiments

### 4.4 Integration Architecture

The button integrates with the NeuralOps platform through a **tRPC‑based API layer** that provides end‑to‑end type safety between the React frontend and the Django backend:

```
Frontend (Zone 4)                    Backend (Django)
─────────────────                    ────────────────
TrainButton.click()
    │
    ├─→ tRPC mutation: createExperiment(config)
    │       │
    │       ├─→ Experiment.objects.create(...)
    │       ├─→ TrainingRun.objects.create(...)
    │       ├─→ scheduler.enqueue(run_id)
    │       └─→ return { experimentId, runId, estimatedStart }
    │
    ├─→ Zustand store: setPendingExperiment(runId)
    │
    ├─→ Toast notification: "Experiment launched!"
    │
    └─→ Router.push(`/experiments/${runId}/monitor`)
```

The tRPC layer is implemented using **Next.js App Router Route Handlers** with **Server‑Sent Events (SSE)** for real‑time progress updates. As the experiment runs, the dashboard can optionally display a **live progress bar** showing training progress without leaving the attention view.

### 4.5 Quick Launch vs. Configure & Launch

Two launch modes cater to different user intents:

| Mode | Behavior | Target User |
|------|----------|-------------|
| **Quick Launch** | Takes all defaults, launches immediately with a single click. Shows a 3‑second countdown with "Undo" option. | Power users who know exactly what they want |
| **Configure & Launch** | Expands the configuration panel with all options editable. Includes cost estimate and duration prediction. | Researchers who want fine‑grained control |

### 4.6 Post‑Launch Experience

After launching an experiment, the dashboard does **not** navigate away. Instead:

1. A **non‑blocking toast** appears: "Experiment launched! [View Live] [Dismiss]"
2. The "Train with This Attention" button transforms into a **progress indicator**: a circular progress ring showing the experiment's status (queued → running → completed)
3. A **side drawer** (accessible via a tab on the right edge of Zone 4) shows a mini‑dashboard with live training metrics (loss curve, accuracy, current epoch)
4. When training completes, a **notification** prompts the user to compare the trained model's attention patterns with the original visualization

### 4.7 Edge Cases & Error Handling

| Scenario | Behavior |
|----------|----------|
| **No GPU available** | The button shows "No GPU Available" in a warning state. Clicking opens the `gpu_marketplace` to rent GPU time. |
| **Insufficient credits** | A modal appears: "You need $2.50 in credits. Current balance: $1.20. [Add Credits] [Cancel]" |
| **Model too large for browser** | For models >2GB, the button redirects to the server‑side launcher (the existing `/launcher` page) with all parameters pre‑filled in the URL |
| **User not authenticated** | The button shows "Sign in to train". Clicking opens the login modal with a return URL back to the current dashboard state. |
| **Network offline** | The configuration is saved to PGlite local database. When connectivity returns, the experiment is automatically submitted. |
| **Experiment fails** | A notification appears with the error message and a "Retry with different config" button that opens the configuration panel with the previous settings preserved. |

---

## 📝 Feature 15: Create Note / Generate Report / Export to Paper

### 5.1 Purpose & Design Philosophy

These three buttons transform attention exploration into **persistent, shareable, and citable research output**. They represent the final stage of the research workflow: **Discover → Analyze → Document → Publish**. Each button serves a different audience and use case:

- **Create Note**: Quick capture for personal reference or team sharing (informal)
- **Generate Report**: Comprehensive documentation for presentations, meetings, or supplementary materials (semi‑formal)
- **Export to Paper**: Publication‑ready output for conferences and journals (formal)

### 5.2 Visual Design

The three buttons are displayed as a **segmented control** (like iOS segmented buttons) — a single row with three segments, where only one can be active at a time:

```
┌──────────────────────────────────────────┐
│  [ 📝 Note ]  [ 📊 Report ]  [ 📄 Paper ] │
└──────────────────────────────────────────┘
```

When a segment is clicked, it expands downward to reveal its specific options, while the other segments collapse. The animation uses Framer Motion's `layoutId` for a smooth shared‑element transition between segments.

### 5.3 Feature 15a: Create Note

#### What It Does
Captures the current attention visualization state (model, layer, head, text, view mode, zoom level, color scheme) plus the rendered attention heatmap (as an SVG) into a `ResearcherNote` in the platform's `notes` app.

#### Integration Architecture
```
Frontend (Zone 4)                          Backend (Django)
─────────────────                          ────────────────
CreateNoteButton.click()
    │
    ├─→ Capture current state from Zustand store
    │       ├── modelId, attentionType, attentionConfig
    │       ├── inputText, tokens
    │       ├── selectedLayer, selectedHead
    │       ├── activeView, zoomLevel, colorScheme
    │       └── computed statistics (entropy, importance, etc.)
    │
    ├─→ Capture SVG from Zone 2 canvas
    │       └── document.querySelector('.head-view-svg').outerHTML
    │
    ├─→ tRPC mutation: createNote({
    │       title: "Attention Analysis — BERT‑base L5‑H7",
    │       note_type: "attention_analysis",
    │       content: markdownContent,     // auto‑generated
    │       metadata: { attentionState, svgData },
    │       tags: ["attention", "bert-base", "galerkin"],
    │       run_id: linkedExperimentId,    // if launched from Feature 14
    │    })
    │
    ├─→ Success toast: "Note created! [View Note]"
    │
    └─→ Update history: mark exploration as "noted"
```

#### Auto‑Generated Markdown Content
The note content is auto‑generated using a template engine that combines the attention state with natural language descriptions. The template structure:

```markdown
# Attention Analysis: BERT‑base‑uncased, Layer 5, Head 7

**Date:** 2026‑05‑26 14:32 UTC
**Model:** BERT‑base‑uncased (110M params)
**Attention Type:** Galerkin (Petrov‑Galerkin projection, rank=64)
**Input Text:** "The quick brown fox jumps over the lazy dog"

## Attention Pattern

[Embedded SVG: attention heatmap]

## Statistics

| Metric | Value |
|--------|-------|
| Entropy | 2.34 bits |
| Concentration | 67.3% |
| Head Importance | 0.87 |
| Head Type | Induction Head (confidence: 0.92) |

## Key Observations
- The head attends strongly to the previous token (diagonal pattern)
- Token "fox" shows high attention to "quick" (adjective‑noun relation)
- Token "jumps" attends to "fox" and "dog" (subject‑verb‑object pattern)

## Comparison
- Similar to Layer 3, Head 2 (cosine similarity: 0.89)
- Distinct from Layer 5, Head 11 (cosine similarity: 0.12)

## Related Experiments
- [Experiment #42]: Trained with this attention configuration (accuracy: 0.873)
- [Experiment #38]: Ablated this head (accuracy dropped by 2.1%)

---
*Generated by NeuralOps Attention Lab · [View Interactive Version](URL)*
```

#### Additional Features
- **@mentions**: Users can tag collaborators in the note (e.g., "@jane look at this induction head!"), which sends a notification via the `messaging` app
- **Inline discussion**: The note automatically creates a thread in the `community` Q&A app, linked bidirectionally
- **Version history**: Every edit to the note creates a `NoteVersion` snapshot, enabling rollback
- **Sharing**: One‑click sharing via public link (the `ShareableLink` system), with optional password and expiration

### 5.4 Feature 15b: Generate Report

#### What It Does
Creates a comprehensive, multi‑section report that combines the current attention visualization with analytics from Zone 5, interpretability insights from Zone 6, and references from the platform's paper library. The report is suitable for:
- Lab meetings and internal presentations
- Supplementary materials for paper submissions
- Grant proposals and progress reports
- Educational materials for teaching

#### Report Structure
The generated report contains six sections, each populated from a different zone or app:

| Section | Source | Content |
|---------|--------|---------|
| **1. Executive Summary** | Auto‑generated by LLM | 2‑3 paragraph summary of findings, written in natural language |
| **2. Model & Configuration** | Zone 1 + Zone 4 | Model details, attention type, hyperparameters, input text |
| **3. Attention Visualization** | Zone 2 | Embedded interactive SVG heatmaps (head view + model view) |
| **4. Quantitative Analysis** | Zone 5 | Entropy, concentration, importance scores, clustering dendrogram, Logit Lens timeline |
| **5. Interpretability Insights** | Zone 6 | Ablation results, circuit tracer analysis, linguistic annotation overlay |
| **6. References & Related Work** | `agents` paper library | Auto‑fetched papers related to the attention type, model, and observed patterns |

#### LLM‑Powered Summary Generation
The Executive Summary is generated by calling an LLM (via the Vercel AI SDK's `streamText`) with a carefully constructed prompt:

```
You are an expert in transformer interpretability. Given the following data about an attention head, write a 2-3 paragraph executive summary suitable for a research report. Include:
- What type of head this is (positional, semantic, induction, etc.)
- What linguistic phenomena it captures
- How it compares to other heads in the same model
- Practical implications (e.g., "This head could be pruned with minimal impact")

Model: BERT-base-uncased
Layer: 5, Head: 7
Attention Type: Galerkin (rank=64)
Input Text: "The quick brown fox jumps over the lazy dog"
Entropy: 2.34
Head Importance: 0.87
Head Type: Induction Head (confidence: 0.92)
...
```

The LLM response streams token‑by‑token into the report preview, using the Vercel AI SDK's `useChat` hook for real‑time streaming.

#### Export Formats
The report can be exported in multiple formats:
- **Interactive HTML**: A self‑contained HTML file with embedded interactive visualizations (using the same React components, compiled to static JS/CSS)
- **PDF**: Rendered server‑side via Puppeteer or browser‑side via `window.print()` with optimized print styles
- **Markdown**: For version control (Git) or import into other tools (Notion, Obsidian)
- **LaTeX**: For direct inclusion in papers (connects to Feature 15c)

### 5.5 Feature 15c: Export to Paper

#### What It Does
Opens the platform's `paper` LaTeX editor with a pre‑written "Methods" and "Results" section containing the attention visualization as a publication‑ready figure. This is the final stage of the research pipeline: from exploration to publication.

#### Pre‑Generated Content

**Methods Section** (auto‑generated):
```latex
\subsection{Attention Analysis}

We analyzed the attention patterns of \texttt{BERT-base-uncased} using the
NeuralOps Attention Visualization Lab. The model was loaded via the
\texttt{transformers} library and the attention weights were extracted for
all 12 layers and 12 heads. The input text ``The quick brown fox jumps over
the lazy dog'' was tokenized using the BERT WordPiece tokenizer, resulting in
10 tokens. Attention patterns were visualized using the BertViz-style head
view, with colors scaled by attention weight magnitude (0.0--1.0).

For the Galerkin attention variant (rank=64), the Petrov-Galerkin projection
was applied to the key and value matrices using a random sketching matrix
$S \in \mathbb{R}^{N \times r}$ drawn from $\mathcal{N}(0, 1/\sqrt{N})$.
```

**Results Section** (auto‑generated):
```latex
\subsection{Results}

Figure~\ref{fig:attention-l5-h7} shows the attention pattern for Layer 5,
Head 7. This head exhibits a strong diagonal pattern characteristic of an
induction head, attending primarily to the immediately preceding token.
The head's entropy of 2.34 bits indicates moderately focused attention,
with 67.3\% of the attention mass concentrated on the top-3 tokens.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/attention_l5_h7.pdf}
  \caption{Attention pattern for BERT-base-uncased, Layer 5, Head 7,
           with Galerkin attention (rank=64).}
  \label{fig:attention-l5-h7}
\end{figure}

When compared to standard multi-head attention, the Galerkin variant
preserves 94.2\% of the attention pattern structure (measured by Jensen-Shannon
divergence), while reducing computational complexity from $O(N^2 \cdot D)$
to $O(N \cdot r \cdot D)$.
```

#### Figure Export Pipeline
The attention visualization is exported as a **publication‑ready PDF figure** through a pipeline:

1. **SVG Capture**: The current Zone 2 SVG is captured at 2× resolution (for print quality)
2. **LaTeX Font Rendering**: Token labels and annotations are re‑rendered using LaTeX fonts (Computer Modern) via a WebAssembly build of MathJax, ensuring the figure matches the paper's typography
3. **PDF Conversion**: The SVG is converted to PDF via `svg2pdf` (a client‑side library) or server‑side via `rsvg-convert`
4. **Figure Caption**: Auto‑generated using the LLM, following academic writing conventions
5. **BibTeX Entry**: The attention type's paper is auto‑cited, with the BibTeX entry fetched from the `agents` paper library

#### Integration with Paper App
The exported content opens directly in the `paper` LaTeX editor (`/paper/editor/{projectId}`):
- If the user has an existing paper project, the sections are inserted at the cursor position
- If not, a new paper project is created with a template (e.g., "ACL 2026 Submission")
- The figures are uploaded as project files and referenced correctly in the LaTeX source

### 5.6 Shared Infrastructure for All Three Buttons

#### Capture Engine
All three buttons share a common **capture engine** that serializes the current dashboard state:

```typescript
interface DashboardCapture {
  // Context
  context: AttentionContext;           // from Zustand store
  
  // Visualizations (SVG strings)
  headViewSvg: string;
  modelViewSvg: string;
  heatmapSvg: string;
  
  // Statistics
  statistics: {
    entropy: number;
    concentration: number;
    importance: number;
    diversity: number;
    headType: string;
    headTypeConfidence: number;
  };
  
  // Ablation results (if Zone 6 is active)
  ablationResults?: {
    ablatedHead: number;
    outputChange: number;
    beforePrediction: string;
    afterPrediction: string;
  };
  
  // Timestamp & attribution
  capturedAt: string;                  // ISO 8601
  capturedBy: string;                  // username
  
  // Links
  explorationId: string;               // for "View Interactive Version"
  experimentId?: string;               // if launched from Feature 14
}
```

#### Performance Considerations
- SVG capture uses `XMLSerializer` which runs synchronously but takes <5ms for typical heatmaps
- For very large matrices (512×512 tokens), SVG capture is deferred to a Web Worker to avoid blocking the main thread
- The capture engine includes a **size estimator** that warns users if the exported SVG exceeds 10MB (common for large models with many tokens)

#### Accessibility
- All three buttons have `aria-label` attributes
- Generated content includes alt text for figures
- Reports include proper heading hierarchy (H1 → H2 → H3) for screen reader navigation
- Color schemes can be switched to colorblind‑friendly palettes before export

---

## 🔗 Cross‑Cutting Concerns

### 6.1 Real‑Time Synchronization

All changes made in Zone 4 are synchronized in real‑time to:
1. **Zone 1 (Context Bar)**: updates the model/attention/layer/head chips
2. **Zone 2 (Visualization Canvas)**: re‑renders the attention view
3. **Zone 3 (Math Panel)**: updates the formula highlighting
4. **Zone 5 (Analytics)**: re‑computes statistics
5. **Zone 6 (Interpretability)**: re‑runs ablation analysis
6. **Zone 7 (Social)**: updates share links and metadata
7. **URL**: updates query parameters for shareability

Synchronization uses Zustand's `subscribe` API with selective subscriptions — each zone only re‑renders when its relevant slice changes, avoiding unnecessary re‑renders.

### 6.2 Offline Support

The Control Panel works offline thanks to:
- **Model caching**: Models loaded via Transformers.js are cached in IndexedDB (up to browser storage limits, typically 2‑4GB)
- **State persistence**: The Zustand store is persisted to `localStorage` via the `persist` middleware
- **Experiment queuing**: If the user clicks "Train with This Attention" while offline, the experiment configuration is stored in PGlite and automatically submitted when connectivity returns
- **Note queuing**: Notes created offline are stored locally and synced when online

### 6.3 Keyboard Shortcuts

Power users can operate Zone 4 entirely via keyboard:

| Shortcut | Action |
|----------|--------|
| `←` / `→` | Previous / Next layer |
| `Shift + ←` / `Shift + →` | Jump 5 layers |
| `↑` / `↓` | Previous / Next head |
| `Ctrl/Cmd + T` | Focus text input |
| `Ctrl/Cmd + Enter` | Quick Launch experiment |
| `Ctrl/Cmd + N` | Create Note |
| `Ctrl/Cmd + R` | Generate Report |
| `Ctrl/Cmd + P` | Export to Paper |
| `Ctrl/Cmd + Z` | Undo last context change |
| `Ctrl/Cmd + Shift + Z` | Redo |

### 6.4 Performance Budget

| Metric | Target |
|--------|--------|
| Layer change → visualization update | < 50ms (perceived as instant) |
| Head change → visualization update | < 50ms |
| Text input → attention re‑computation | < 500ms (debounced) |
| Thumbnail generation (per thumbnail) | < 16ms (Web Worker) |
| "Train with This Attention" → experiment created | < 200ms |
| Note creation → saved to backend | < 500ms |
| Report generation → preview visible | < 2s (streaming) |
| Paper export → LaTeX editor opens | < 1s |

---

## 7. Summary

Zone 4 is the **mission control** of the Attention Dashboard. Its three core features — the Unified Selector, the "Train with This Attention" button, and the Create Note/Report/Paper trio — provide a complete workflow from exploration to publication:

| Feature | Core Innovation | Key Technology |
|---------|----------------|----------------|
| **Unified Selector** | Live thumbnail previews on layer slider; visual head grid with size‑encoded importance | Comlink Web Workers for thumbnail generation; Framer Motion for animations; @tanstack/react-virtual for performance |
| **Train with This Attention** | One‑click bridge from visualization to experiment; pre‑filled launcher with cost estimation | tRPC for type‑safe API calls; SSE for real‑time progress; PGlite for offline queuing |
| **Create Note / Report / Paper** | Three‑tier output: informal note → semi‑formal report → formal paper; LLM‑powered summarization | Vercel AI SDK for streaming LLM generation; MathJax WASM for LaTeX font rendering; svg2pdf for figure export |

Together, these features transform the attention dashboard from a passive visualization tool into an **active research platform** where every insight can be immediately tested, documented, and published — all without leaving the browser.