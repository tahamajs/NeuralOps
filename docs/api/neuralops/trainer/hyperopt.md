Module neuralops.trainer.hyperopt
=================================

Functions
---------

`quick_optimize(train_fn, eval_fn, n_trials=10, study_name='quick_opt') ‑> dict[str, typing.Any]`
:   Quick optimize.

Classes
-------

`HyperparameterOptimizer(train_fn: <class 'collections.abc.Callable'>, eval_fn: <class 'collections.abc.Callable'>, config_template: dict[str, typing.Any], study_name: str = 'svd_attention_opt', storage: str | None = None, direction: str = 'maximize')`
:   

    ### Descendants

    * neuralops.trainer.hyperopt.PrunedHyperparameterOptimizer

    ### Methods

    `get_importance(self) ‑> dict[str, float]`
    :   Return the importance.

    `get_trial_dataframe(self) ‑> pandas.DataFrame`
    :   Return the trial dataframe.

    `objective(self, trial) ‑> float`
    :   Objective.

    `plot_optimization_history(self, save_path=None) ‑> Any`
    :   Plot optimization history.

    `plot_parallel_coordinate(self, save_path=None) ‑> Any`
    :   Plot parallel coordinate.

    `plot_param_importances(self, save_path=None) ‑> Any`
    :   Plot param importances.

    `run(self, n_trials: int = 20, timeout: int | None = None, n_jobs: int = 1, show_progress_bar: bool = True) ‑> dict[str, typing.Any]`
    :   Execute the run operation and return results.

`PrunedHyperparameterOptimizer(*args, pruner='median', **kwargs)`
:   

    ### Ancestors (in MRO)

    * neuralops.trainer.hyperopt.HyperparameterOptimizer