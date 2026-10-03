# 📐 Zone 3: Live Math Panel — 3Blue1Brown‑Style Complete Implementation Specification

## Overview

Zone 3 is the **mathematical nerve center** of the attention dashboard. Occupying a dedicated panel (collapsible, default width 400–480px on desktop, full‑width bottom sheet on mobile), it renders the actual LaTeX equations for attention computation, highlights variables in sync with the visualization canvas, animates step‑by‑step computations, exposes raw numerical values, and supports architectural comparison. Its design is directly inspired by **3Blue1Brown’s** “Attention in transformers, visually explained” video (built with Manim) and the **Self Attention Flow** interactive tool, but taken further into a fully interactive, explorable, and deeply integrated component.

The panel operates on a **bidirectional binding** with all other zones: hovering a token pair in Zone 2’s Head View highlights the corresponding variables in the equation and shows the exact numerical computation path in Zone 3’s Inspector; changing a parameter (e.g., rank in Galerkin attention) in Zone 4 updates the equation and the numerical tables in Zone 3 instantly.

---

## 1. Feature 10: Live LaTeX Equation Renderer with Animated Steps

### 1.1 Purpose & UX Philosophy

Researchers (especially those new to attention mechanisms) often struggle to map the abstract formula to the actual computations happening inside the model. The Live LaTeX Renderer bridges this gap by **rendering the attention formula in a beautiful, annotated LaTeX display** where each variable is color‑coded and linked to the corresponding visualization element. Furthermore, the user can click any part of the formula to trigger an **animated step‑by‑step walkthrough** of the computation, using the currently loaded data (model, layer, head, tokens). The goal is to make the math *alive*—a dynamic, explorable artifact rather than a static image.

### 1.2 Primary Equation Display

For standard scaled dot‑product attention, the main formula displayed is:

```
Attention(Q, K, V) = softmax( QK^T / √d_k ) V
```

Each component is rendered as an interactive LaTeX sub‑expression:

| Component | Color Code | Description |
|-----------|------------|-------------|
| **Q** | Blue (#3b82f6) | Query matrix |
| **K** | Red (#ef4444) | Key matrix |
| **V** | Green (#22c55e) | Value matrix |
| **dₖ** | Gray (#94a3b8) | Head dimension |
| **softmax(…)** | Purple (#a855f7) | Softmax function |
| **√dₖ** | Orange (#f59e0b) | Scaling factor |
| **QK^T** | Teal (#14b8a6) | Dot‑product scores |

These colors are consistent across all views: the Q/K Joint View uses the same blue/red for its scatter points, the Sankey view uses them for node borders, etc.

### 1.3 Variable Highlighting via Hover

When the user hovers over a token pair (i, j) in Zone 2’s Head View, the Live Math Panel responds within 16ms:

- The **Q** and **K** sub‑expressions gently pulse (scale 1.05 with a 300ms easing).
- A small **tooltip** appears near the equation, showing the numeric indices: “Q[3] · K[7]”.
- The **QK^T** part is underlined with a subtle animated wavy line to indicate that this product is being computed for that specific pair.
- The **softmax** part briefly highlights when the hovered cell’s weight has been computed.

This reactive highlighting transforms the static formula into a *dynamic legend* of the attention process.

### 1.4 Step‑by‑Step Animated Computation

Clicking on any part of the equation opens the **Computation Stepper**—a modal or inline panel (depending on viewport size) that breaks the attention calculation into discrete, animated steps:

#### Step 1: Matrix Multiplication (Q × K^T)
- The LaTeX equation shows: `Scores = Q K^T`
- The actual Q and K matrices for the selected layer/head are displayed as **mini heatmaps** (using the same color scale as Zone 2) side‑by‑side.
- An animation shows the two matrices sliding together, with the dot‑product values appearing cell‑by‑cell in the resulting Scores matrix.
- The user can **scrub** a timeline slider to rewind/fast‑forward the animation.
- The Q, K, and Scores matrices are rendered as small, labeled heatmaps (size proportional to the number of tokens, max 200×200px).

#### Step 2: Scaling by √dₖ
- Equation: `ScaledScores = Scores / √dₖ`
- The numeric value of √dₖ (e.g., √64 = 8) is displayed.
- An animation shows the Scores matrix being uniformly scaled down—the heatmap colors shift, and the numerical values update.
- A comparison view (toggleable) shows the Scores matrix before and after scaling, with a difference heatmap.

#### Step 3: Softmax (Row‑by‑Row)
- Equation: `AttentionWeights = softmax(ScaledScores)`
- The softmax is animated **row by row**: the user can select which row (query token) to watch, or let the animation cycle through all rows.
- For the selected row:
  - The raw scaled scores are shown as a bar chart.
  - One by one, the bars transform: exponentiate (eˣ) → sum → normalize.
  - The resulting attention weights are displayed as a final bar chart and also as percentages.
- The animation uses a **glow‑and‑fade** effect on each step, with a speed control (0.25× to 4×).

#### Step 4: Weighted Sum (× V)
- Equation: `Output = AttentionWeights × V`
- The attention weights (row vector) and the Value matrix (columns) are displayed.
- The animation shows each Value vector being multiplied by its corresponding attention weight, then summed to produce the output vector for that query token.
- The final output vector is displayed, and optionally compared to the actual hidden state (from the model) to show the impact of the residual connection (though that’s beyond the attention formula, it’s a nice educational touch).

### 1.5 Scrubber & Playback Controls

The Computation Stepper includes:
- **Play/Pause** button
- **Step Forward/Backward** (skip to next/previous major step)
- **Micro‑step Forward/Backward** (advance within a step, e.g., exponentiate just one more score)
- **Timeline Slider** (scrub through the entire computation)
- **Speed Selector** (0.25×, 0.5×, 1×, 2×, 4×)
- **Loop Toggle** (loop the entire animation)
- **Auto‑Advance Toggle** (automatically proceed to next step after a configurable delay)

### 1.6 Multi‑Head and Multi‑Layer Support

When the user has selected multiple heads (via Zone 4 checkboxes or the Model View shift‑click), the Live Math Panel adapts:
- The Q, K, V matrices shown are the **concatenated or averaged** versions (user‑selectable).
- The computation steps show the **per‑head** contributions before aggregation.
- A small head‑indicator tab appears above the matrices, allowing the user to see individual head computations.

### 1.7 Variant Equations for Different Attention Types

The dashboard supports many attention types (Galerkin, TLinFormer, PADRe, Spectraformer, RPC, FUSIONATTN, etc.), each with its own mathematical formulation. The Live Math Panel automatically switches the displayed formula based on the selected attention type. For example:

- **Galerkin Attention**:
  ```
  K_sketch = K·S,   V_sketch = V·S
  Scores = Q·K_sketch^T / √d
  Attention = softmax(Scores)
  Output = Attention · V_sketch
  ```
  The formula shows the random sketching matrix **S** (with its dimensions `N × r` indicated) and highlights how the complexity reduces from `O(N²)` to `O(N·r)`.

- **PADRe (Polynomial Attention)**:
  ```
  Scores = QK^T / √d
  Base = 1 + Scores / p
  AttentionWeights = (Base)^p  (clamped and normalized)
  Output = AttentionWeights · V
  ```
  The polynomial order `p` is highlighted, and the animation step shows the polynomial evaluation instead of softmax.

- **TLinFormer**:
  ```
  K_proj = E·K,   V_proj = F·V
  Scores = Q·K_proj^T / √d
  Attention = softmax(Scores)
  Output = Attention · V_proj
  ```
  The low‑rank projection matrices **E** and **F** are displayed with their dimensions.

Each variant is accompanied by a short textual description and a link to the original paper (pulled from the attention registry). The equation rendering uses a **unified expression tree**, allowing shared components (like `Scores`, `softmax`) to be rendered consistently while swapping out the outer structure.

### 1.8 Architecture Comparison Math Diff (Link with Feature 12)

When the user is in comparison mode (two attention types selected), the math panel splits vertically (or stacks on narrow screens). Both formulas are displayed side‑by‑side, with **diff highlighting**:
- Components that are **identical** between the two formulas are rendered in normal opacity.
- Components that **differ** (e.g., `softmax` vs. `polynomial`, or presence/absence of a projection matrix) are highlighted with a glow and a subtle “changed” badge.
- A unified step‑by‑step animation can run **both formulas simultaneously**, allowing the user to see how the computation diverges at each step.

---

## 2. Feature 11: Numerical Value Inspector

### 2.1 Purpose & Design

The Numerical Value Inspector is a **live, scrollable data table** that shows the exact floating‑point values at each stage of the attention computation for the currently selected token pair. It’s the “debugger” of the attention mechanism—researchers can trace exactly which numbers led to a particular attention weight. This feature is inspired by the “watch” panels in scientific computing environments like MATLAB, Mathematica, and the PyTorch debugger.

### 2.2 Table Structure

The Inspector table has a **hierarchical row layout**, organized by computation step:

```
┌──────────────────────────────────────────────────────┐
│ COMPUTATION INSPECTOR — Token: 'The' (i=3) → 'cat' (j=7) │
├──────────────────────────────────────────────────────┤
│ ① Q·K Dot Product                                   │
│   Q[3] vector norm: 3.42                             │
│   K[7] vector norm: 2.89                             │
│   Q[3] · K[7] = 0.847 (raw score)                   │
│                                                      │
│ ② Scale by √dₖ (√64 = 8.000)                        │
│   0.847 / 8.000 = 0.106 (scaled score)               │
│                                                      │
│ ③ Softmax                                            │
│   Row 3 scaled scores: [0.106, 0.034, 0.215, ...]    │
│   e^(0.106) = 1.112                                  │
│   Σ e^(row) = 48.731                                 │
│   1.112 / 48.731 = 0.023 (attention weight)          │
│                                                      │
│ ④ Weighted Value                                     │
│   V[7] vector (first 5 dims): [0.12, -0.34, ...]     │
│   0.023 × V[7] = [0.003, -0.008, ...]                │
│                                                      │
│ FINAL: attention weight = 0.0228                     │
└──────────────────────────────────────────────────────┘
```

Each row group is collapsible. The currently active row (based on the animation step) is highlighted with a blue left border. The table uses **monospace font** (`JetBrains Mono` or `Fira Code`) for numerical values, with **right alignment** for readability.

### 2.3 Live Updates

The Inspector updates **instantly** when:
- The user hovers over a different cell in the Head View (new token pair → new computation path).
- The user scrubs through the Computation Stepper (values update to reflect the current step).
- The user changes the layer, head, or attention type (entire table recomputes).
- The user modifies attention parameters (temperature, rank, etc.) — the values re‑compute with the new parameters.

### 2.4 Color Coding of Values

To make the table scannable, values are color‑coded:
- **Positive large** (≥ 0.5): green (#22c55e)
- **Positive small** (0 to 0.5): gray (#94a3b8)
- **Zero**: faint (#64748b)
- **Negative**: red (#ef4444)
- **Extreme** (≥ 10 or ≤ ‑10): bold with orange highlight (#f59e0b)

The attention weight (the final value) is displayed in a slightly larger font size and in bold.

### 2.5 Token Pair Selection Methods

The Inspector can display values for:
- **A single token pair** (hovered or clicked in Head View) — default mode.
- **All token pairs for a selected query row** (user clicks a query token → table shows the entire softmax row with all key values).
- **Top‑N pairs** (sorted by attention weight) for a selected head.
- **Custom pair** (user types token indices manually).

### 2.6 Export Capabilities

- **Copy row**: A copy icon next to each row copies that row’s values to clipboard.
- **Export CSV**: A button exports the entire Inspector data for the current selection as a CSV file.
- **Export LaTeX table**: For inclusion in papers, a button generates a LaTeX `tabular` environment with the values.
- **Pin values**: The user can “pin” a specific computation (e.g., the softmax row for token 3) so it remains visible even when hovering elsewhere. Pinned rows appear in a separate “Pinned” section at the top of the Inspector.

### 2.7 Performance Considerations

For large matrices (e.g., 512 tokens), computing and displaying values for all pairs would be overwhelming. The Inspector uses **lazy evaluation**: only the values needed for the currently selected token pair (or top‑N pairs) are computed. The underlying attention data is stored in a Float32Array for fast random access.

---

## 3. Feature 12: Architecture Comparison Math Diff

### 3.1 Purpose & Core Design

This feature allows researchers to compare the mathematical formulations of two attention mechanisms side‑by‑side. It is an extension of the Live LaTeX Renderer that renders **two formula panels** and visually highlights the differences. This is critical for understanding the trade‑offs between attention types (e.g., standard MHA vs. Galerkin) and for educational purposes (showing how approximations modify the original formula).

### 3.2 Side‑by‑Side Layout

When comparison mode is activated (via a toggle in Zone 4 or by selecting “Compare” in the Model Selector), the math panel splits into two columns:

```
┌──────────────────────────────┬──────────────────────────────┐
│ STANDARD MHA                 │ GALERKIN (r=64)              │
│                              │                              │
│ S = QK^T / √d               │ K' = K·S  (sketch)          │
│ A = softmax(S)               │ V' = V·S                    │
│ O = A·V                      │ S' = Q·K'^T / √d            │
│                              │ A' = softmax(S')             │
│                              │ O' = A'·V'                   │
└──────────────────────────────┴──────────────────────────────┘
```

### 3.3 Visual Diff Mechanism

Differences between the two formulas are highlighted using a **unified diff model** (like `git diff` but for math):

- **Lines present in Left but not Right**: shaded red on the left, with a `-` marker.
- **Lines present in Right but not Left**: shaded green on the right, with a `+` marker.
- **Lines that are identical**: displayed normally (gray text on both sides).
- **Lines that are similar but with different parameters**: highlighted in yellow, with the differing parameter emphasized (e.g., `√d` vs. `√d_new`).

### 3.4 Interactive Parameter Linking

When a parameter differs between the two formulas (e.g., the rank `r` in Galerkin, or the polynomial order `p` in PADRe), the user can **adjust that parameter via a slider** in the comparison panel. As the slider moves, the right‑hand formula updates in real time, and the corresponding numerical values in the Inspector recompute. This enables interactive “what‑if” analysis: “What happens to the attention weights if I increase the Galerkin rank from 32 to 128?”

### 3.5 Synchronized Step‑by‑Step Animation

The Computation Stepper can run **both formulas simultaneously**, with the animation steps synchronized. For example:

- Step 1 (Left): `S = QK^T` | Step 1 (Right): `K' = K·S` (these are fundamentally different, so the animation diverges).
- Step 2 (Left): `A = softmax(S)` | Step 2 (Right): `S' = Q·K'^T`.
- Step 3 (Left): `O = A·V` | Step 3 (Right): `A' = softmax(S')`.
- Step 4 (Right): `O' = A'·V'`.

The animation timeline shows markers where the two computations **diverge** and **re‑converge** (e.g., both produce an output matrix of the same shape).

### 3.6 Complexity & Performance Annotations

Below each formula, a compact table shows the computational complexity (Big‑O notation) and estimated FLOPs for the current sequence length and dimensions. For example:

| Metric | Standard MHA | Galerkin (r=64) |
|--------|--------------|-----------------|
| Time Complexity | O(N²·D) | O(N·r·D) |
| Memory | O(N²) | O(N·r) |
| Est. FLOPs (N=128) | 2.1M | 0.3M |
| Speedup | 1× | 7× |

This table updates dynamically when the user changes the sequence length (via a test‑sentence input) or the attention parameters. It directly links to the **benchmark leaderboard** (Zone 5) where actual measured performance data is available.

### 3.7 “Learn More” Deep Links

Each attention type in the comparison view has a **“Learn More”** button that opens a side panel with:
- The original paper abstract (fetched from the `agents` paper library or Semantic Scholar).
- Key equations from the paper.
- Links to implementations (GitHub, HuggingFace).
- Community discussions (from the `community` Q&A app).
- Example usage in experiments (from the `experiment` app).

---

## 4. Additional Advanced Modules for Zone 3

### Module A: Interactive Matrix Visualization (Mini‑Heatmaps)

Alongside the LaTeX formula, small **heatmap thumbnails** of the Q, K, V, and attention matrices are displayed. These are not just static images—they are **live and interactive**:

- Hovering over a cell in a mini‑heatmap shows the value in a tooltip.
- Clicking a mini‑heatmap expands it to fill the panel (like a picture‑in‑picture).
- The mini‑heatmaps are synchronized with the Head View: selecting a region in the Head View highlights the corresponding rows/columns in the Q/K/V mini‑heatmaps.

This gives the researcher a **direct visual mapping** from the abstract matrices in the formula to the concrete attention weights they see in Zone 2.

### Module B: Animated Proof/Explanation of the Attention Mechanism

A dedicated “Explain” mode that walks the user through **why attention works**, not just how. This uses a combination of:
- Animated illustrations (e.g., tokens talking to each other).
- Textual explanations at each step (e.g., “The dot product measures how ‘compatible’ a query is with a key. If Q and K are similar, their dot product is large, meaning this token pair should have high attention.”)
- Interactive quizzes at the end of each section.

This transforms Zone 3 into a **built‑in mini‑course** on attention mechanisms, reducing the barrier to entry for new researchers.

### Module C: Computational Graph View

Instead of (or in addition to) the linear step‑by‑step view, the user can switch to a **computational graph** representation. The attention computation is shown as a **DAG (Directed Acyclic Graph)** where nodes are operations (MatMul, Scale, Softmax) and edges are data tensors. Each node shows:
- Operation name
- Input/output shapes
- FLOP count
- Live status (completed, in‑progress, error)

Clicking a node highlights the corresponding sub‑expression in the LaTeX formula. This is inspired by TensorBoard’s graph visualization and is especially useful for understanding complex attention variants like FUSIONATTN, which combine multiple operations.

### Module D: Parameter Sensitivity Analysis

A panel where the user can adjust key parameters (e.g., temperature, rank, polynomial order, number of heads) via sliders and see **real‑time effects** on:
- The LaTeX formula (parameters update inline).
- The numerical values in the Inspector.
- The attention heatmap in Zone 2 (via the ViewSynchronizer).
- The FLOPs estimate and complexity table.

A **sensitivity plot** (generated by running multiple forward passes with varying parameters) can be displayed, showing how attention entropy, concentration, or output similarity changes as a function of the parameter. This enables researchers to tune attention mechanisms for their specific needs.

### Module E: Citation & Reference Panel

A collapsible panel that pulls in **relevant literature** for the currently displayed attention type. It shows:
- The paper title, authors, and year.
- The abstract (truncated to 3 lines with “Read more”).
- Key mathematical definitions from the paper.
- BibTeX citation that can be copied with one click.
- Links to the paper on arXiv, Semantic Scholar, and the user’s own library (if saved via the `agents` paper library).
- “Related papers” section based on citation graph or semantic similarity.

This is integrated with the `agents` paper search and library, so researchers can immediately save a paper to their collection or annotate it.

### Module F: Export to Notebook / LaTeX

The entire math panel content—including the LaTeX formula, computation steps, numerical values, and mini‑heatmaps—can be exported as:
- **Jupyter Notebook (.ipynb)**: A Python notebook that reproduces the attention computation using PyTorch/TensorFlow code, with the current parameters and data. This allows researchers to take their exploration directly into code.
- **LaTeX Document**: A compilable LaTeX file with the equations, tables, and embedded figures (SVG heatmaps), suitable for inclusion in a paper. This connects to the `paper` app for further editing and compilation.

### Module G: Voice‑Over Narration

An accessibility and educational feature: the step‑by‑step animation can be accompanied by **auto‑generated audio narration** (using Web Speech API or a TTS service). The narration reads out the explanation for each step. The user can select the narration language (English, Chinese, Spanish, etc.) and speed. This makes the math panel accessible to visually impaired users and auditory learners.

### Module H: Gamified Challenge Mode

To make learning attention mechanisms more engaging, a **Challenge Mode** can be activated. The system presents the user with a partially completed computation (e.g., a softmax row with one value missing) and asks them to predict the missing value. Points are awarded for accuracy. Challenges can be:
- **Timed**: Complete as many predictions as possible in 60 seconds.
- **Precision‑based**: Score based on how close the prediction is to the actual value.
- **Head‑type identification**: Given an attention heatmap, identify whether the head is positional, semantic, etc.

High scores are saved to the user’s profile and contribute to the gamification system (Zone 8).

---

## 5. Integration with Other Zones

Zone 3 is tightly coupled with the rest of the dashboard:

| Zone | Integration |
|------|-------------|
| **Zone 1 (Navigation)** | The context bar’s model/attention type selection updates the formula displayed in Zone 3. |
| **Zone 2 (Visualization)** | Hover/select events in the Head View highlight variables and update the Numerical Inspector. The Computation Stepper can be triggered from Zone 2’s context menu (“Explain this cell”). |
| **Zone 4 (Control Panel)** | Parameter sliders (temperature, rank, etc.) update the equation and numerical values in real time. |
| **Zone 5 (Analytics)** | The FLOPs table and complexity annotations are linked to the analytics panel for more detailed profiling. |
| **Zone 7 (Social)** | The “Share” button generates a shareable link that includes the current math panel state (selected attention type, parameters, computation step). |

---

## 6. Performance Considerations

- **LaTeX rendering**: Use KaTeX (faster than MathJax) for initial render. Pre‑compile the equation templates for each attention type and cache the DOM.
- **Animation performance**: The computation animations use CSS transforms and `requestAnimationFrame` for smooth 60fps. Heavy computations (e.g., recomputing attention for parameter sweeps) are offloaded to a Web Worker.
- **Numerical values**: Store attention tensors as `Float32Array` for fast random access without re‑computing the entire matrix for each hover.
- **Memory**: Mini‑heatmaps are rendered as small canvas elements (< 50KB each). Only the currently visible matrices are kept in memory.

---

## 7. Accessibility

- **Keyboard navigation**: The Computation Stepper can be controlled entirely via keyboard (Space to play/pause, Left/Right arrows to step, numbers to jump to step).
- **Screen reader**: All LaTeX formulas have `aria-label` attributes with plain‑text descriptions (e.g., “Attention equals softmax of Q times K transpose over square root of d_k, times V”). The Numerical Inspector table uses proper `<table>` semantics with `<caption>` and `<th>` elements.
- **Color contrast**: The formula colors meet WCAG AA contrast ratios against the dark background. The diff highlighting uses both color and symbols (‑/+) for colorblind users.
- **Reduced motion**: The `prefers-reduced-motion` media query disables all animations and shows static computation steps instead.

---

## 8. Internationalization

All textual explanations (step descriptions, tooltips, labels) are externalized into locale files and can be translated. The LaTeX formulas themselves are language‑independent, but the surrounding explanatory text adapts to the user’s locale. Initial supported languages: English, Chinese, Spanish, Japanese, Arabic (right‑to‑left layout for the math panel is supported).

---

## 9. Summary

Zone 3 is not just a math display—it is a **fully interactive, animated, and deeply integrated computational environment** that makes the abstract mathematics of attention tangible and explorable. With features like live LaTeX highlighting, step‑by‑step computation animations, numerical inspection, architecture comparison diffs, and additional modules (mini‑heatmaps, computational graphs, sensitivity analysis, export, narration, gamification), it serves both novice learners and expert researchers. It is the bridge between the visual (Zone 2) and the analytical (Zone 5), and it turns the dashboard into a comprehensive platform for attention mechanism research.