Module neuralops.analysis.utils
===============================
Utility functions for analysis.

Functions
---------

`correlation_matrix(df: pandas.DataFrame, columns: list[str]) ‑> pandas.DataFrame`
:   Correlation matrix.

`detect_least_compressible_layers(model, tokenizer, calibration_texts: list[str], k: int = 5) ‑> list[int]`
:   Detect least compressible layers.

`filter_results(df: pandas.DataFrame, models: list[str] | None = None, tasks: list[str] | None = None, ranks: list[int] | None = None, min_accuracy: float | None = None) ‑> pandas.DataFrame`
:   Filter results.

`get_best_configurations(df: pandas.DataFrame, metric: str = 'accuracy') ‑> pandas.DataFrame`
:   Return the best configurations.

`statistical_tests(df: pandas.DataFrame, group_col: str, value_col: str) ‑> dict[str, typing.Any]`
:   Statistical tests.