Module neuralops.analysis.model_comparison
==========================================
Multi‑run model comparison and plotting.

Functions
---------

`quick_compare(results_dirs: list[str], output_dir: str = './results/comparison') ‑> None`
:   Quick compare.

Classes
-------

`ModelComparison(results_dirs: list[str], output_dir: str = './results/comparison')`
:   

    ### Methods

    `generate_all_plots(self) ‑> None`
    :   Generate all plots.

    `load_all_results(self) ‑> pandas.DataFrame`
    :   Load all results from disk or config.