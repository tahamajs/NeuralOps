# 📚 SVD Linear Attention Research Hub & Documentation

This directory contains the **Interactive Research Hub SPA**, **184 Academic Module Specifications**, **Architecture Comparison Studio**, and **Master Empirical Reports**.

---

## 🌟 Structure & Components

```
Docs/
├── index.html                  # Single Page Application (SPA) Research Hub
├── app.js                      # Client-side routing, KaTeX pre-rendering, Comparison Studio
├── style.css                   # Responsive dark/light theme design system
├── catalog.json                # Complete catalog of all 184 documentation articles
├── Attention_Visualization_Dashboard/  # 9-Zone Attention Canvas specifications
├── Mapping_.../                # 15 Transformer Component Exploration Studio specs
├── content/                    # All 184 publication-grade documentation markdown files
│   ├── attentions/             # 104 Attention mechanism academic articles
│   ├── compressions/           # 80 Model compression engine academic articles
│   └── framework/              # Master synthesis compendiums
├── reports/                    # Empirical Benchmark Reports (JSON, CSV, Markdown)
└── results/                    # Master benchmark summaries
```

---

## 🌐 Deploying to GitHub Pages

To serve the Research Hub on GitHub Pages:
1. Go to repository **Settings > Pages**.
2. Under **Build and deployment > Source**, select **Deploy from a branch**.
3. Choose branch: `main`, folder: `/Docs`.
4. Click **Save**.

The Research Hub will be live at `https://<username>.github.io/<repository>/`.

---

## 🎛️ Interactive Architecture Comparison Studio

Click the **Compare** button on the navigation bar in `index.html` to:
- Select up to 4 models simultaneously (e.g. *SVD Linear Attention*, *FlashAttention-3 FP8*, *MambaSVD-Flash*, *Dense MHA*).
- Inspect Big-$\mathcal{O}$ complexity formulas.
- Compare WikiText-2 / C4 Perplexity scores.
- Compare multi-sequence latency profiles ($L=512, 2048, 8192$ tokens).
- Compare peak VRAM footprints, speedup ratios, and Triton hardware acceleration status.
