# 🧭 Zone 1: Global Navigation & Context Bar — Complete Implementation Specification

## Overview

Zone 1 is the **persistent header** that spans the full width of the dashboard at the very top of the viewport (typically 56–64px height on desktop, 48–56px on mobile). It serves three critical functions simultaneously: (1) it **informs** the researcher exactly what they are looking at at all times, (2) it enables **rapid switching** between models, attention types, layers, heads, and input texts without losing context, and (3) it provides **spatial orientation** through breadcrumb navigation plus a temporal trail of past explorations. The bar must remain **visible at all times** (position: sticky, top: 0, z-index: 50) regardless of how far the user scrolls into the visualization canvas or analytics panels below.

The header follows an **inverted‑L navigation pattern** widely used in complex dashboards and research platforms: global navigation links on the left, contextual controls in the center, and user/account actions on the right. This is the same pattern employed by the New York Times, SAP Analytics Cloud, and enterprise observability platforms like Grafana. 

---

## 1. Feature 1: Persistent Model/Text Context Header

### 1.1 Purpose & UX Philosophy

When a researcher is deep in layer 7, head 3 of a Galerkin‑attention variant of BERT‑base, analyzing a specific input sentence, they must **never** wonder "Which model am I looking at?" or "What text is this?" This information must be immediately visible, glanceable, and **always accurate**. Research shows that dashboards lacking context "merely present data; they do not provide meaningful insights" — context bars improve "context observability by making the currently active dataset/model state explicit in the UI".

### 1.2 Visual Layout & Anatomy

The context header occupies the **center section** of the global navigation bar, between the breadcrumb trail (left) and the user menu (right). It displays as a horizontal chain of **labeled chips/badges** separated by subtle dividers (vertical pipe or chevron). Each chip is independently clickable and hoverable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🏠 Home  >  Attention Lab  >  BERT-base  │  Attention: Galerkin (r=64) │ │
│                                         │  Text: "The quick brown..."  │ │
│                                         │  Layer 5  │  Head 7          │ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Chip design specifications:**
- **Background**: `bg-white/5` (glass effect), with a `border border-white/10` for definition.
- **Padding**: `px-3 py-1.5` for comfortable click targets (minimum 32px height for accessibility).
- **Typography**: The label portion (e.g., "Model:") uses `text-[10px] font-bold text-slate-500 uppercase tracking-widest`. The value portion (e.g., "BERT-base-uncased") uses `text-sm font-semibold text-white`.
- **Dividers**: A subtle `│` character or a 1px vertical line in `border-white/10`.
- **Hover state**: `hover:bg-white/10 transition-colors duration-150`.
- **Active/Click state**: A brief scale animation (`active:scale-95`) provides tactile feedback.

### 1.3 Individual Chip Behaviors

#### Model Chip
- **Display format**: `Model: BERT-base-uncased` or `Model: GPT-2 (fine-tuned)`.
- **Click behavior**: Opens a **compact dropdown** listing the user's recently used models (last 5), models from their `model_versioning` registry, and a search input that queries HuggingFace Hub. This is the same model selector described in Feature 2, but embedded inline for quick access.
- **Tooltip**: Hovering shows the full model name, parameter count, and last used date.
- **Truncation**: If the model name exceeds 25 characters, it is truncated with an ellipsis (…), and the full name appears in the tooltip.

#### Attention Type Chip
- **Display format**: `Attention: Galerkin (rank=64)` or `Attention: Standard MHA (12 heads)`.
- **Click behavior**: Opens a dropdown of **all registered attention types** from the `ATTENTION_REGISTRY` (Galerkin, TLinFormer, PADRe, Spectraformer, RPC, FUSIONATTN, etc.). Each option shows a one‑line description (e.g., "Galerkin: Petrov‑Galerkin projection with random sketching, O(N·r·D)").
- **Visual indicator**: The attention type chip has a **colored left border** (3px) that matches the attention type's designated color in the system. For example, Galerkin = blue, TLinFormer = green, PADRe = purple. This color persists throughout the dashboard, creating a consistent visual language.

#### Text Input Chip
- **Display format**: `Text: "The quick brown fox jumps over..."` (truncated to 30 characters).
- **Click behavior**: Expands into an **inline text input** (replacing the chip with an `<input>` field) where the user can type or paste any text. Pressing Enter triggers re‑analysis. The input supports up to 512 tokens, with a character counter.
- **Recent texts**: A small dropdown arrow reveals the last 10 analyzed texts, stored in `localStorage` per user.

#### Layer & Head Chips
- **Display format**: `Layer 5` and `Head 7` as two separate chips, visually grouped together with a lighter background.
- **Interaction**: Each chip is a **stepper control** — clicking reveals tiny `‑` and `+` buttons that appear on hover, allowing quick increment/decrement without opening the full control panel. For power users, the arrow keys (← →) change the layer, and (↑ ↓) change the head when the visualization canvas is focused.
- **Visual feedback**: When the user changes the layer via the Zone 4 slider, the Layer chip updates **instantly** with a subtle pulse animation (scale 1.05 for 200ms) to draw attention to the change.

### 1.4 Responsive Behavior

| Viewport Width | Behavior |
|----------------|----------|
| ≥ 1280px (Desktop) | All chips displayed inline with full labels |
| 1024–1279px (Tablet Landscape) | Labels shortened: "Model:" → icon only, values preserved |
| 768–1023px (Tablet Portrait) | Chips collapse into a single dropdown: "BERT-base · Galerkin · Layer 5 · Head 7" with a dropdown arrow. Click expands all chips vertically. |
| < 768px (Mobile) | Only the most critical info shown: model name + layer/head. Full context accessible via a "Details" button that opens a bottom sheet. |

### 1.5 Performance Considerations

- **State management**: The context state (model, attention type, text, layer, head) is stored in a **Zustand store** (`useAttentionContext`). All visualization components subscribe to this store. When any value changes, only the affected components re‑render.
- **URL synchronization**: Every context change updates the URL query parameters (`?model=bert-base&attn=galerkin&layer=5&head=7&text=...`). This makes every exploration **shareable via URL**. On page load, the URL is parsed to restore the previous state.
- **Debouncing**: Text input changes are debounced at 500ms to avoid excessive re‑computation while the user is typing.

### 1.6 Accessibility

- All chips are `<button>` elements with `aria-label` attributes (e.g., `aria-label="Current model: BERT-base-uncased. Click to change."`).
- The context bar is reachable via the **Tab key** as the first focusable element on the page (after a "Skip to content" link).
- Screen readers announce context changes via `aria-live="polite"` regions.

---

## 2. Feature 2: Quick‑Switch Model Selector

### 2.1 Purpose & Design Philosophy

Researchers need to rapidly compare attention patterns across different models and attention mechanisms. The Quick‑Switch Model Selector enables changing the model **without leaving the dashboard** — the entire visualization updates reactively. This is inspired by Neuronpedia's approach where "you can now host Neuronpedia yourself, with your own model and data" and HuggingFace's model discovery interface, adapted for the specific needs of attention visualization comparison.

### 2.2 Architecture & Component Tree

```
ModelSelector (parent container)
├── AttentionTypeSelector (dropdown)
├── ModelSourceTabs (3 tabs: "Your Models" | "HuggingFace Hub" | "Upload")
├── SearchInput (text input with autocomplete)
├── ModelList (scrollable list of model cards)
│   └── ModelCard (individual model entry)
│       ├── ModelIcon (architecture indicator)
│       ├── ModelName + Version
│       ├── ModelMeta (parameter count, last used, attention types supported)
│       └── QuickActions (⭐ Favorite, 📊 Compare, ℹ️ Info)
└── SelectedModelPreview (confirmation footer)
```

### 2.3 The Dropdown Trigger

The dropdown trigger is integrated into the Model Chip in the context bar (see Feature 1). Additionally, a dedicated **"Change Model"** button sits in Zone 4 (Control Panel). Both open the same dropdown.

**Trigger appearance:**
- Default state: The model chip (described above).
- When dropdown is open: The chip gains a `ring-2 ring-neon-blue/50` border and a subtle chevron rotation animation.

### 2.4 Dropdown Panel Anatomy

The dropdown is a **floating panel** (not a native `<select>`) rendered via a portal to avoid z‑index issues. It appears directly below the trigger with a smooth scale+fade animation (200ms, `transform-origin: top`).

**Dimensions**: 480px wide × 520px max‑height on desktop. On mobile, it becomes a full‑screen bottom sheet.

**Internal structure (top to bottom):**

#### Tab Bar: "Your Models" | "HuggingFace Hub" | "Upload"
- Three tabs that filter the model source.
- Active tab has a `border-b-2 border-neon-blue` indicator.
- "Your Models" shows models from the user's `model_versioning` registry and recent experiments.
- "HuggingFace Hub" queries the HF Hub API in real time.
- "Upload" opens a file picker for local model weights.

#### Search Bar
- A text input with a magnifying glass icon on the left.
- Autocomplete suggestions appear after 3 characters typed.
- Debounced at 300ms for HF Hub queries.
- Supports filtering by: model name, architecture type, parameter count range, task type (classification, generation, etc.).

#### Model List
- Virtualized list (using `react-window` or `@tanstack/react-virtual`) to handle hundreds of models smoothly.
- Each row is a **ModelCard** component (described below).
- Infinite scroll for HuggingFace results (loads 20 at a time).
- Loading state shows skeleton cards (shimmer effect).

#### ModelCard Design
```
┌──────────────────────────────────────────────────────────┐
│ [🧠]  BERT-base-uncased                   ⭐  📊  ℹ️    │
│       110M params · Used 3 days ago                      │
│       Supported: MHA, Galerkin, PADRe, TLinFormer        │
│       Tasks: Classification, NER, QA                      │
│       ───────────────────────────────────────────────     │
│       [Select]  [Compare with current]                   │
└──────────────────────────────────────────────────────────┘
```

**Card anatomy:**
- **Left icon**: Indicates model architecture (🧠 = Transformer, 👁️ = Vision, etc.).
- **Model name**: Bold, white text. Truncated with ellipsis if needed.
- **Meta line**: Gray text showing parameter count, last used date.
- **Supported attention types**: Small colored badges for each supported attention mechanism.
- **Tasks**: Small tags for task compatibility.
- **Action buttons**: "Select" (primary, blue) and "Compare with current" (secondary, outline). The "Compare" option opens a side‑by‑side view with the currently loaded model.

#### Selected Model Preview Footer
When a model is selected but not yet confirmed, a sticky footer appears at the bottom of the dropdown:
```
┌──────────────────────────────────────────────────────────┐
│ Selected: Galerkin (rank=64) · BERT-base-uncased         │
│ This will re‑compute attention for the current text.     │
│ [Cancel]                              [Apply & Visualize]│
└──────────────────────────────────────────────────────────┘
```

### 2.5 Attention Type Quick‑Switch

Next to the model selector, a separate dropdown (or integrated tab) lists all registered attention types. Each option shows:

```
┌──────────────────────────────────────────────────────┐
│ ● Galerkin Attention                                 │
│   Petrov‑Galerkin projection with random sketching   │
│   Complexity: O(N·r·D) · Rank: 64                    │
│   Paper: https://arxiv.org/abs/2105.14906            │
│   ───────────────────────────────────────────────     │
│ ● TLinFormer                                         │
│   Low‑rank projections for K and V                   │
│   Complexity: O(N·r·D) · Rank: 64                    │
│   ───────────────────────────────────────────────     │
│ ● PADRe (Polynomial Attention)                       │
│   Polynomial kernel: (1 + S/p)^p                     │
│   Order: 2 · Temperature: 1.0                        │
│   ───────────────────────────────────────────────     │
│ ...                                                  │
└──────────────────────────────────────────────────────┘
```

Each attention type entry includes:
- A **colored dot** matching the attention type's system color.
- **Name** (bold).
- **One‑line description** of the mechanism.
- **Complexity** notation.
- **Configurable parameters** (rank, order, temperature) that can be adjusted inline.
- **Link to paper/arXiv** (external link icon).

### 2.6 Loading States & Transitions

When a new model is selected:
1. The dropdown closes with a fade animation.
2. The Model Chip in the context bar shows a **pulsing skeleton** (shimmer effect) for 200–500ms while the model loads.
3. All visualization panels (Zone 2) show a **loading overlay** with a progress indicator: "Loading BERT‑base weights… (34MB / 440MB)".
4. If the model is already cached (in IndexedDB or browser cache), loading is near‑instantaneous with a success toast: "Model loaded in 0.3s".
5. If loading fails, an error toast appears: "Failed to load model. Check your connection or try a different model." with a retry button.

### 2.7 Performance Optimizations

- **Model caching**: Loaded models are cached in IndexedDB (via `transformers.js` or a custom caching layer). Subsequent loads of the same model are instant.
- **Lazy loading**: Attention types are lazy‑loaded. Only the currently selected attention type's computation runs.
- **Web Worker**: Heavy attention computation (matrix multiplication, softmax) runs in a Web Worker to avoid blocking the main thread.
- **Optimistic UI**: The context bar updates immediately (optimistic), then the visualization catches up. If the computation fails, the context bar reverts.

### 2.8 Accessibility

- The dropdown is fully keyboard‑navigable: Arrow keys move between options, Enter selects, Escape closes.
- `aria-expanded` on the trigger button, `aria-selected` on the active option, `role="listbox"` on the dropdown.
- Focus is trapped within the dropdown while it's open.

---

## 3. Feature 3: Breadcrumb Navigation with History

### 3.1 Purpose & UX Philosophy

Researchers exploring attention patterns often follow a non‑linear path: they might start with BERT‑base, switch to a fine‑tuned variant, compare two layers, jump to a different attention type, and then want to return to a previous view. The breadcrumb trail provides **spatial orientation** (where am I in the exploration?), while the history panel provides **temporal navigation** (how did I get here?).

This dual‑navigation system follows the pattern established by data analysis tools where "each 'crumb' in the breadcrumb path UI control may indicate a summary of a respective data query and/or filter submitted over the course of the current data analysis path" — enabling users to "jump from any first" step to any later step instantly.

### 3.2 Breadcrumb Trail Design

#### Visual Layout
The breadcrumb trail occupies the **leftmost section** of the navigation bar. It displays as a horizontal chain of clickable segments separated by chevron arrows (› or /).

```
🏠 Home  ›  Attention Lab  ›  BERT-base  ›  Layer 5  ›  Head 7
```

#### Crumb Anatomy
Each crumb is composed of:
- An **optional icon** (home icon for the first crumb).
- A **text label** (truncated to 20 characters max).
- A **separator** (chevron `›` or forward slash `/`).

#### Crumb States
| State | Appearance | Behavior |
|-------|------------|----------|
| **Default** | `text-slate-400 hover:text-white` | Click navigates to that level |
| **Current (last crumb)** | `text-white font-semibold` | Not clickable (already here) |
| **Truncated (middle)** | `text-slate-400` with `…` | Click expands to show hidden crumbs |
| **Hover** | `text-neon-blue underline` | Shows tooltip with full path |

#### Responsive Truncation
When the breadcrumb trail exceeds the available width:
1. Middle crumbs are replaced with a `…` button.
2. Clicking `…` expands a dropdown showing the hidden crumbs.
3. On mobile (< 768px), only the current crumb is shown, with a back arrow (←) that navigates up one level.

#### Breadcrumb Levels
The breadcrumb hierarchy follows the natural attention exploration path:

```
Level 1: Home (dashboard)
Level 2: Attention Lab (the attention dashboard itself)
Level 3: Model name (e.g., "BERT-base-uncased")
Level 4: Layer number (e.g., "Layer 5")
Level 5: Head number (e.g., "Head 7")
```

Additional optional levels:
- **Attention Type**: Inserted between Level 2 and Level 3 if the user explicitly selects a non‑default attention type.
- **Comparison Mode**: A special crumb "Compare: BERT vs GPT‑2" appears when in comparison view.
- **Text Preview**: The first 15 characters of the input text can appear as a crumb if the text was explicitly set.

### 3.3 History Panel Design

#### Trigger
A **clock icon** (🕐) with a badge showing the number of history entries sits at the right edge of the breadcrumb area. Clicking it opens the history panel.

#### Panel Anatomy
The history panel is a **dropdown/floating panel** (320px wide, 480px max‑height) that appears below the breadcrumb.

```
┌──────────────────────────────────────────┐
│ 📜 Exploration History           [Clear] │
├──────────────────────────────────────────┤
│ ● BERT-base / Layer 5 / Head 7          │
│   Galerkin (r=64) · 2 min ago           │
│   ───────────────────────────────────    │
│ ○ GPT-2 / Layer 3 / Head 2              │
│   Standard MHA · 8 min ago              │
│   ───────────────────────────────────    │
│ ○ BERT-base / Layer 5 / Head 3          │
│   PADRe (order=2) · 15 min ago          │
│   ───────────────────────────────────    │
│ ○ BERT-base / All Layers (Model View)   │
│   TLinFormer (r=32) · 22 min ago        │
│   ───────────────────────────────────    │
│ ...                                     │
└──────────────────────────────────────────┘
```

#### History Entry Anatomy
Each history entry shows:
- A **colored dot** indicating the attention type.
- The **model name**, **layer**, and **head** (or view type).
- The **attention type** and its configuration parameters.
- A **relative timestamp** ("2 min ago", "1 hour ago").
- A **preview thumbnail** (tiny heatmap preview, 48×48px) generated at the time of the exploration. This thumbnail is stored in `localStorage` as a base64 data URL.

#### Interactions
- **Click** an entry: Restores that exact exploration state (model, attention type, layer, head, text, view mode, zoom level). The entire dashboard updates to match.
- **Star (⭐)** an entry: Saves it to "Saved Explorations" (persisted to the backend via the `notes` system). Saved explorations appear at the top of the history panel with a gold border.
- **Compare (📊)** : Opens the selected entry in a side‑by‑side comparison view with the current exploration.
- **Delete (✕)** : Removes the entry from history.
- **Clear All**: Removes all history entries (with confirmation dialog).

#### History Persistence
- **localStorage**: The last 50 explorations are stored in `localStorage` per browser. This survives page refreshes and tab closes.
- **Backend (optional)** : If the user is authenticated, history entries are synced to the backend via the `activity` feed, enabling cross‑device history.
- **Export**: Users can export their history as JSON or CSV for external analysis.

### 3.4 Back/Forward Navigation

In addition to the breadcrumb and history panel, the browser's native **Back** and **Forward** buttons are fully supported. Each context change pushes a new entry to the browser history stack via `history.pushState()`. This means:

- Pressing the browser Back button returns to the previous attention exploration.
- Pressing Forward returns to the later exploration.
- The URL is always shareable and bookmarkable.

### 3.5 Related Explorations Panel

At the bottom of the history panel, a **"Related Explorations"** section appears. This uses collaborative filtering to suggest:

- **"Other researchers who viewed this pattern also explored…"**
- **"Similar patterns"** (based on attention pattern similarity metrics).
- **"Top‑starred patterns with this model"**.

Each suggestion is a clickable card that loads that exploration directly.

### 3.6 Edge Cases & Error States

| Scenario | Behavior |
|----------|----------|
| **Empty history** (first‑time user) | Shows a friendly illustration: "Your exploration history will appear here. Start by selecting a model and text!" with a CTA button. |
| **History entry references a deleted model** | The entry is grayed out with a "Model no longer available" badge. The preview thumbnail remains visible. |
| **localStorage full** | Oldest entries are automatically evicted. A notification appears: "Your oldest exploration history has been cleared to save space." |
| **Browser in incognito mode** | History is session‑only. A banner at the top of the panel notes: "History is temporary in this session." |
| **Corrupted history entry** | Skipped gracefully. The entry shows "⚠️ Could not restore this exploration" and can be deleted. |

### 3.7 Accessibility for History

- The history panel is opened via a button with `aria-label="Exploration history"` and `aria-expanded`.
- Each history entry is a `<button>` with `aria-label="Restore exploration: BERT-base Layer 5 Head 7, Galerkin attention, 2 minutes ago"`.
- The clear history button requires confirmation via a modal dialog to prevent accidental data loss.

---

## 4. Integration Specifications

### 4.1 How Zone 1 Communicates with Other Zones

Zone 1 acts as the **single source of truth** for the current exploration context. All other zones subscribe to this state:

```
Zone 1 (Context Bar)
    │
    ├── Updates ──→ Zone 2 (Visualization Canvas) — re‑renders attention view
    │               Zone 3 (Math Panel) — re‑renders equations with new values
    │               Zone 4 (Control Panel) — syncs sliders and selectors
    │               Zone 5 (Analytics) — re‑computes statistics
    │               Zone 6 (Interpretability) — re‑runs ablation analysis
    │               Zone 7 (Social) — updates share links and metadata
    │               Zone 8 (Gamification) — triggers XP events
```

The communication is handled through:
1. **Zustand store** (`useAttentionContext`) for synchronous state.
2. **URL query parameters** for shareability and browser history.
3. **WebSocket** (optional) for collaborative sessions — when one user changes the context, it syncs to others.

### 4.2 State Contract

The context state object follows this TypeScript interface:

```typescript
interface AttentionContext {
  // Model
  modelId: string;               // e.g., "bert-base-uncased"
  modelSource: 'registry' | 'huggingface' | 'upload';
  modelVersion?: string;         // from model_versioning

  // Attention
  attentionType: string;         // e.g., "galerkin", "padre"
  attentionConfig: Record<string, any>;  // e.g., { rank: 64, order: 2 }

  // Text
  inputText: string;
  tokens: string[];

  // Navigation
  selectedLayer: number;
  selectedHead: number;
  activeView: 'head' | 'model' | 'heatmap' | '3d' | 'sankey' | 'qk';

  // Comparison (optional)
  compareModelId?: string;
  compareAttentionType?: string;

  // Metadata
  lastModified: number;          // timestamp
  explorationId: string;         // unique ID for this exploration
}
```

### 4.3 Performance Budget

| Metric | Target |
|--------|--------|
| Context bar initial render | < 16ms (one frame) |
| Model switch response time (cached) | < 100ms to update context bar |
| Model switch response time (uncached) | < 3s with progress indicator |
| URL update after context change | < 50ms |
| History panel open | < 100ms |
| Breadcrumb update | < 16ms |

### 4.4 Testing Checklist

- [ ] Context bar remains sticky when scrolling to bottom of analytics panel
- [ ] All chips are keyboard‑accessible (Tab, Enter, Escape)
- [ ] Screen reader announces context changes
- [ ] URL updates correctly on every context change
- [ ] Browser Back/Forward correctly restores previous states
- [ ] Model switching works when offline (cached models)
- [ ] History persists across page refreshes
- [ ] Mobile responsive: chips collapse correctly
- [ ] Color contrast ratios meet WCAG AA standards (4.5:1 for text)
- [ ] No layout shift when chips update (stable dimensions)
- [ ] History panel handles 500+ entries without performance degradation
- [ ] Breadcrumb correctly truncates on narrow viewports
- [ ] Compare mode breadcrumb appears correctly
- [ ] Attention type colors are consistent across all zones
- [ ] "Clear history" confirmation dialog works
- [ ] Starred explorations sync to backend
- [ ] Related explorations load without blocking the UI
- [ ] Error states (model load failure) are handled gracefully
- [ ] First‑time user sees appropriate empty states

---

## 5. Summary

Zone 1 is the **orchestrator** of the entire attention dashboard. It provides three essential navigation functions:

| Feature | Core Value | Key Innovation |
|---------|------------|----------------|
| **Persistent Context Header** | Never lose track of what you're looking at | Live‑editable chips that serve as both display and control |
| **Quick‑Switch Model Selector** | Change models without leaving the dashboard | Integrated attention type selector with complexity annotations and paper links |
| **Breadcrumb + History** | Navigate spatially and temporally | Dual navigation with starred explorations, comparison mode, and cross‑device sync |

Together, these three features ensure that researchers can explore attention patterns across dozens of models and attention mechanisms without ever feeling lost — while maintaining the ability to backtrack, compare, and share their discoveries at any moment.