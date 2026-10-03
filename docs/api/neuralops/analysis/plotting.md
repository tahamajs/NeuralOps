Module neuralops.analysis.plotting
==================================
Comprehensive plotting functions for model analysis.

Functions
---------

`create_comprehensive_plots(df: pandas.DataFrame, output_dir: str, figsize: tuple[int, int] = (18, 16)) ‑> None`
:   Build and return comprehensive plots.

`plot_accuracy_curve(history, metrics, config, output_dir, opts) ‑> Any`
:   Plot accuracy curve.

`plot_accuracy_curves(history, metrics, config, output_dir, opts) ‑> Any`
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

`plot_loss_curves(history, metrics, config, output_dir, opts) ‑> Any`
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

`set_style() ‑> None`
:   Set the style.