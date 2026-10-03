Module neuralops.analysis.generate_sweep_publication_artifacts
==============================================================
Publication-Grade Sweep Analysis, Reporting, and Visualization Engine.

Generates:
1. High-resolution publication figures (both 300-DPI PNG and vector PDF for LaTeX):
   - fig1_top_modules_ranked.png / .pdf
   - fig2_accuracy_vs_runtime.png / .pdf (with Pareto frontier)
   - fig3_status_breakdown.png / .pdf
   - fig4_category_performance.png / .pdf
2. Publication-ready LaTeX tables (paper_tables.tex) with booktabs formatting.
3. Detailed Markdown report (comprehensive_report.md).
4. Structured full-results JSON export (comprehensive_results.json).
5. Optional continuous --watch mode for live updating while sweeps run.

Functions
---------

`classify_module(name: str, path: str = '') ‑> tuple[str, str]`
:   Classify module name into primary type and fine-grained architectural sub-family.

`generate_latex_tables(df: pandas.DataFrame, out_file: pathlib.Path)`
:   Generate publication-ready LaTeX tables using booktabs.

`generate_markdown_report(df: pandas.DataFrame, out_file: pathlib.Path, output_dir: pathlib.Path)`
:   Generate extensive publication-ready markdown report.

`load_and_enrich_summary(summary_csv: pathlib.Path) ‑> pandas.DataFrame`
:   Read summary.csv and enrich with classifications and derived metrics.

`main()`
:   

`plot_fig1_top_modules(df: pandas.DataFrame, out_dir: pathlib.Path, top_n: int = 25)`
:   Plot top performing modules vs full baseline.

`plot_fig2_accuracy_vs_runtime(df: pandas.DataFrame, out_dir: pathlib.Path)`
:   Plot Accuracy vs Runtime / Latency Pareto Frontier.

`plot_fig3_status_breakdown(df: pandas.DataFrame, out_dir: pathlib.Path)`
:   Plot Status Breakdown (PASS, INCOMPATIBLE, SKIP, FAIL) by Module Type.

`plot_fig4_category_performance(df: pandas.DataFrame, out_dir: pathlib.Path)`
:   Plot Mean Accuracy and Variance across Architectural Sub-families.