Module neuralops.experiment_utils
=================================
Django integration layer – background experiment runner,
WebSocket broadcaster, AI assistant, and multi‑run comparison.

Sub-modules
-----------
* neuralops.experiment_utils.comparison
* neuralops.experiment_utils.config
* neuralops.experiment_utils.data
* neuralops.experiment_utils.main
* neuralops.experiment_utils.model
* neuralops.experiment_utils.orchestrator
* neuralops.experiment_utils.runner
* neuralops.experiment_utils.tasks
* neuralops.experiment_utils.train

Functions
---------

`broadcast_terminal(msg: str) ‑> None`
:   Send a log message to the terminal WebSocket group.

`compare_runs_api(run_ids: list[str], baseline_id: str | None = None) ‑> dict`
:   API-friendly wrapper.

`run_compression(config, output_dir) ‑> Any`
:   Run compression.

`run_evaluation(config, model_path) ‑> Any`
:   Run evaluation.

`run_training(config, stop_event=None, progress_callback=None, log_callback=None, final_callback=None) ‑> Any`
:   Run training.

`run_training_async(experiment, stop_event: threading.Event) ‑> None`
:   Main entry point called by Django in a background thread.
    
    Args:
        experiment: Django model instance with fields:
            id, config (JSON), status, logs, current_step, total_steps, etc.
        stop_event: threading.Event to signal cancellation.

Classes
-------

`AIAssistant(llm_function)`
:   Monitors training metrics and suggests actions (e.g., early stop).
    Uses a callable LLM function for decision making.

    ### Methods

    `monitor(self, exp_id: str, metrics_history: list) ‑> None`
    :   Monitor.

`ExperimentComparator(run_ids: list[str], baseline_id: str | None = None)`
:   Compare multiple TrainingRuns with rich diff output.
    Inspired by ZenML's comparison tool, MLflow compare, and RagaAI diff.

    ### Methods

    `compare(self) ‑> neuralops.experiment_utils.comparison.ComparisonResult`
    :   Compare.

    `load_runs(self) ‑> None`
    :   Load runs from disk or config.

`MegaOrchestrator(config: dict[str, typing.Any], run_id: str, base_dir: pathlib.Path, progress_callback: <class 'collections.abc.Callable'> | None = None, log_callback: <class 'collections.abc.Callable'> | None = None, final_callback: <class 'collections.abc.Callable'> | None = None, stop_event: threading.Event | None = None)`
:   Central orchestrator that connects all framework modules.
    
    Usage (from Django runner):
        orchestrator = MegaOrchestrator(
            config=request.data,
            run_id=str(uuid.uuid4()),
            base_dir=Path(settings.MEDIA_ROOT),
            progress_callback=progress_cb,
            log_callback=log_cb,
            final_callback=final_cb,
            stop_event=stop_event,
        )
        orchestrator.run()

    ### Methods

    `run(self) ‑> dict[str, typing.Any]`
    :   Execute the full experiment pipeline.