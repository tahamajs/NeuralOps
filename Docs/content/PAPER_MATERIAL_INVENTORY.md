# 📚 NeuralOps & SVD-Lin Paper Material Inventory & Asset Compilation
**Target Conferences**: NeurIPS 2026 / ICLR 2026  
**Compiled**: 2026-08-26  
**Scope**: All LaTeX manuscripts, empirical results, mathematical documentation, figures, and benchmark telemetry.

---

## 1. LaTeX Manuscripts & Draft Sections

The repository hosts two complete, high-quality LaTeX manuscript codebases in `paper/`:

### A. Manuscript 1: `NeuralOps` (Comprehensive Universal Benchmark & Hardware Co-Design)
* **Directory**: `paper/neuralops_paper/`
* **Entry Point**: `paper/neuralops_paper/main.tex`
* **Preamble & Styles**: `paper/neuralops_paper/preamble.tex`

| File Path | Description | Key Excerpts / Contents |
| :--- | :--- | :--- |
| `paper/neuralops_paper/sections/00_abstract.tex` | Complete 250-word Abstract | Mathematical framing of $O(N^2)$ to $O(N)$ linear attention, HBM arithmetic intensity bottlenecks, 200+ modules, fused SRAM kernels, and Stiefel manifold optimization. |
| `paper/neuralops_paper/sections/01_introduction.tex` | Introduction & Core Motivations | Context window scaling challenges, memory hierarchy breakdown, and the 4 central contributions of NeuralOps. |
| `paper/neuralops_paper/sections/02_background.tex` | Theoretical Foundations | Multi-head attention formulation, linear attention duality ($Q(K^\top V)$ vs $(QK^\top)V$), and SSM recurrent scans. |
| `paper/neuralops_paper/sections/03_related_work.tex` | Literature Review | Detailed taxonomy: Low-rank (Linformer, SVD), SSMs (Mamba, S4, Griffin), MoE hybrids, and kernel approximations. |
| `paper/neuralops_paper/sections/04_methodology.tex` | Algorithmic Formulation | Full mathematical derivation of adaptive SVD, Stiefel Riemannian manifold updates ($\operatorname{St}(p, n)$), and singular entropy dynamic rank allocation. |
| `paper/neuralops_paper/sections/05_systems_roofline.tex` | Systems & Roofline Model | Operational intensity ($I = \text{FLOPs}/\text{Bytes}$), GPU SRAM vs HBM memory access analysis, and Triton kernel profiling. |
| `paper/neuralops_paper/sections/06_experimental_evaluation.tex` | Empirical Results & Analysis | GLUE benchmark evaluation, LRA long-sequence scaling, memory efficiency frontiers, and ablation studies. |
| `paper/neuralops_paper/sections/07_broader_impacts.tex` | Broader Impacts & Ethics | Energy efficiency, carbon footprint reduction ($CO_2\text{e}$ via CodeCarbon), and edge-device accessibility. |
| `paper/neuralops_paper/sections/08_conclusion.tex` | Concluding Remarks & Future Work | Synthesis of linear attention scaling, manifold-constrained optimization, and future long-context frontiers. |
| `paper/neuralops_paper/appendices/app_00_overview.tex` to `app_05_proofs.tex` | 6 Detailed Appendices | Exhaustive module catalog, mathematical proofs of convergence, error bounds for Eckart-Young low-rank projection, and hyperparameter tables. |

---

### B. Manuscript 2: `SVD-Lin` (Data-Dependent Orthonormal Low-Rank Linear Attention)
* **Directory**: `paper/svd_lin_paper/`
* **Entry Point**: `paper/svd_lin_paper/main.tex`

| File Path | Description | Key Excerpts / Contents |
| :--- | :--- | :--- |
| `paper/svd_lin_paper/sections/00_abstract.tex` | Abstract | Focuses on SVD-Lin data-dependent orthonormal projections vs fixed random projections (Linformer), achieving $>99.5\%$ BERT accuracy with $2.3\times$ speedup. |
| `paper/svd_lin_paper/sections/01_introduction.tex` | Introduction | Motivation for optimal Frobenius-norm sequence compression via Eckart-Young theorem. |
| `paper/svd_lin_paper/sections/02_background.tex` | Background | Self-attention mechanics, computational bottlenecks, and low-rank approximation theory. |
| `paper/svd_lin_paper/sections/03_methodology.tex` | SVD-Lin Derivation | Dual-projection formulation, orthonormal basis extraction, and online rank adaptation. |
| `paper/svd_lin_paper/sections/04_experimental_setup.tex` | Experimental Setup | GLUE benchmark configuration, model baselines (BERT-base, RoBERTa), and hardware parameters. |
| `paper/svd_lin_paper/sections/05_results.tex` | Results & Performance Tables | Accuracy vs rank $r \in \{64, 128, 256\}$, throughput scaling, and memory savings. |
| `paper/svd_lin_paper/sections/06_discussion.tex` | Discussion & Error Analysis | Spectral decay profiles, singular value cutoff thresholds, and approximation stability. |
| `paper/svd_lin_paper/sections/07_conclusion.tex` | Conclusion | Summary of low-rank linear attention and practical deployment guidelines. |

---

## 2. Quantitative Results & Telemetry Data

| File Path | Format | Description & Metrics Contained |
| :--- | :---: | :--- |
| `results/gpu_server/snapshot_2026_08_26/benchmark_summary.csv` | CSV | $380+$ combinations with forward/backward latency, VRAM allocation, throughput, and GLUE validation accuracy. |
| `results/master_benchmark_summary.csv` | CSV | Aggregated multi-track benchmark results across all 5 paradigms (SVD, SSM, MoE, Kernel, Pruning). |
| `results/retest_heavy_modules/gpu_retest_summary.json` | JSON | GPU verification log for heavy modules (`factorised_matrix_state`, `mamba_svd_flash`, `gated_delta_net`, `hgrn2_layer`, etc.). |
| Live API (`/api/stats` / `/api/export/latex-table`) | HTTP / JSON | Real-time live data engine providing dynamic NeurIPS booktabs LaTeX tables and module leaderboards. |

---

## 3. Mathematical Documentation & Algorithmic Reference (`Docs/`)

| File Path | Category | Key Insights for Paper |
| :--- | :--- | :--- |
| `Docs/content/PROJECT_FULL_AUDIT.md` | Architectural Audit | Deep mathematical formulations for all 45 core modules, complexity tables, and failure forensics. |
| `Docs/content/MODULE_OPTIMIZATION_ADVICE.md` | Empirical Recommendations | Concrete latency/memory trade-offs and recommendations across sequence lengths $L \in [128, 64000]$. |
| `Docs/content/attentions/*.md` ($125+$ files) | Module Reference Cards | Per-module forward/backward mathematical equations, FLOP formulas, and parameter footprints. |
| `Docs/content/compressions/*.md` ($63+$ files) | Compression Formulations | Detailed singular value thresholding, structured pruning, and quantization formulas. |

---

## 4. Analysis & Export Scripts

| Script Path | Purpose | Output Formats |
| :--- | :--- | :--- |
| `Codes/framework/run_quality_sweep.py` | Multi-Seed Quality Engine | Computes Bootstrap 95% CIs, Benjamini-Hochberg FDR p-values, and validation fidelity. |
| `Codes/framework/visualization/chart_generator.py` | Publication Visualization Studio | Generates 300 DPI Pareto frontiers, radar charts, loss trajectories, and correlation heatmaps. |
| `Codes/framework/live_monitor.py` | Live Dashboard & LaTeX Generator | Generates complete `\begin{table} ... \end{table}` LaTeX markup from results. |
| `prepare_submission.py` | Dual-Conference Packager | Automated double-blind code sanitization and zip packaging for NeurIPS and ICLR. |

---

## 5. Gap Analysis & Next Steps for Paper Finalization

1. **Phase 1 $\rightarrow$ Phase 2 Empirical Transition**:
   - Current numbers in `snapshot_2026_08_26/benchmark_summary.csv` represent Phase 1 (Efficiency & Profiling).
   - Once Phase 1 completes ($720/720$), Phase 2 will supply the **multi-seed convergence metrics** (3 seeds, 95% Bootstrap CI, Benjamini-Hochberg $p$-values) to directly populate Table 1 and Table 2 in `06_experimental_evaluation.tex`.
2. **Figure Generation**:
   - Run `chart_generator.py` on the final Phase 2 dataset to produce vector PDFs (`pareto_frontier.pdf`, `top5_radar.pdf`, `accuracy_heatmap.pdf`) for inclusion in the LaTeX paper.
3. **Double-Blind Anonymization**:
   - Run `prepare_submission.py --target iclr` or `--target neurips` to automatically package the supplementary bundle without author identity leaks.
