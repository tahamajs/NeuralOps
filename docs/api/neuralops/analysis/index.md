Module neuralops.analysis
=========================
Analysis tools: plotting, metrics, layerwise analysis, model comparison, reporting.

Sub-modules
-----------
* neuralops.analysis.comparison
* neuralops.analysis.generate_all_modules_analysis
* neuralops.analysis.generate_sweep_publication_artifacts
* neuralops.analysis.layerwise_analyzer
* neuralops.analysis.metrics
* neuralops.analysis.model_comparison
* neuralops.analysis.plotting
* neuralops.analysis.report_generator
* neuralops.analysis.utils

Functions
---------

`aggregate_results(results_list: list[dict[str, typing.Any]]) ‑> dict[str, typing.Any]`
:   Aggregate results.

`compute_metrics(predictions: numpy.ndarray, labels: numpy.ndarray, task: str = 'classification') ‑> dict[str, float]`
:   Compute and return metrics.

`correlation_matrix(df: pandas.DataFrame, columns: list[str]) ‑> pandas.DataFrame`
:   Correlation matrix.

`create_comprehensive_plots(df: pandas.DataFrame, output_dir: str, figsize: tuple[int, int] = (18, 16)) ‑> None`
:   Build and return comprehensive plots.

`detect_least_compressible_layers(model, tokenizer, calibration_texts: list[str], k: int = 5) ‑> list[int]`
:   Detect least compressible layers.

`estimate_intrinsic_dimension(hidden_states: np.ndarray, k: int = 5) ‑> float`
:   Estimates the intrinsic dimension of a representation manifold using k-NN distances.
    
    Args:
        hidden_states: Sample representation array of shape [N, D].
        k: Neighbor rank for distance estimation.
    
    Returns:
        float: Estimated intrinsic dimension between 1.0 and D.

`filter_results(df: pandas.DataFrame, models: list[str] | None = None, tasks: list[str] | None = None, ranks: list[int] | None = None, min_accuracy: float | None = None) ‑> pandas.DataFrame`
:   Filter results.

`generate_full_report(df: pandas.DataFrame, output_dir: str, experiment_name: str = 'experiment') ‑> str`
:   Generate full report.

`generate_html_report(df: pandas.DataFrame, output_dir: str, title: str = 'Analysis Report') ‑> str`
:   Generate html report.

`generate_interactive_report(df: pandas.DataFrame, output_dir: str) ‑> str`
:   Generate interactive report.

`generate_latex_table(df: pandas.DataFrame, output_path: str, caption: str = 'Results') ‑> str`
:   Generate latex table.

`get_best_configurations(df: pandas.DataFrame, metric: str = 'accuracy') ‑> pandas.DataFrame`
:   Return the best configurations.

`plot_accuracy_curve(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot accuracy curve.

`plot_attention_heatmap(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot attention heatmap.

`plot_confusion_matrix(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot confusion matrix.

`plot_gpu_memory(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot gpu memory.

`plot_gradient_flow(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot training time.

`plot_loss_curve(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot loss curve.

`plot_lr_schedule(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot lr schedule.

`plot_parameter_distribution(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot parameter distribution.

`plot_pareto_front(df, output_dir, x_metric='speedup_factor', y_metric='accuracy') ‑> None`
:   Plot pareto front.

`plot_radar_chart(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot radar chart.

`plot_rank_allocation(recommended_ranks, output_dir) ‑> None`
:   Plot rank allocation.

`plot_tradeoff(df, output_dir, metric='accuracy') ‑> None`
:   Plot tradeoff.

`plot_training_history(history: dict, save_path: str) ‑> None`
:   Plot training history.

`plot_training_time(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot training time.

`quick_compare(results_dirs: list[str], output_dir: str = './results/comparison') ‑> None`
:   Quick compare.

`set_style() ‑> None`
:   Set the style.

`statistical_tests(df: pandas.DataFrame, group_col: str, value_col: str) ‑> dict[str, typing.Any]`
:   Statistical tests.

Classes
-------

`LayerwiseAnalyzer(model: nn.Module, tokenizer: Any, device: str | torch.device | None = None)`
:   Streaming Layerwise Analyzer that captures inter-layer representations,
    outlier distribution spikes, and effective rank without accumulating VRAM memory.

    ### Methods

    `analyze(self, texts: list[str], batch_size: int = 8) ‑> dict[str, typing.Any]`
    :   Executes streaming layerwise analysis on input text corpus.
        
        Args:
            texts: List of input string prompts/sentences.
            batch_size: Batch size for tokenization and model inference.
        
        Returns:
            Dict containing cosine similarities, effective ranks, outlier statistics,
            and least-compressible layer rankings.

    `register_streaming_hooks(self) ‑> None`
    :   Registers streaming forward hooks on all model transformer layers.

    `remove_hooks(self) ‑> None`
    :   Remove all active hooks and free memory.

`MetricsTracker(filepath: str)`
:   Tracks per‑step metrics and saves as JSON lines.

    ### Methods

    `log(self, metrics: dict[str, typing.Any], step: int | None = None, flush: bool = False) ‑> None`
    :   Log.

    `save(self) ‑> None`
    :   Save.

    `to_dataframe(self) ‑> pandas.DataFrame`
    :   To dataframe.

`ModelComparison(results_dirs: list[str], output_dir: str = './results/comparison')`
:   

    ### Methods

    `generate_all_plots(self) ‑> None`
    :   Generate all plots.

    `load_all_results(self) ‑> pandas.DataFrame`
    :   Load all results from disk or config.