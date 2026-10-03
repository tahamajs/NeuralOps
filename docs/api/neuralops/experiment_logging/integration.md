Module neuralops.experiment_logging.integration
===============================================
Automatic integration between logging, ResultsSaver, and the experiment runner.
Call `auto_setup(run_config)` once at the start of training.

Functions
---------

`auto_setup(run_config: dict, run_id: str, base_dir: pathlib.Path) ‑> dict`
:   Create logger and tracker from run config, returning them as a dict.
    Also writes config snapshot to the run directory.
    
    Args:
        run_config: dictionary containing keys such as `log_level`, `log_format`,
                    `save_logs`, `exp_name`, etc.
        run_id: unique identifier for this run.
        base_dir: root directory where run artifacts are stored.
    
    Returns:
        A dictionary with keys `logger`, `tracker`, and `config`.