Module neuralops.experiment_utils.orchestrator
==============================================
MegaOrchestrator – single entry point that executes the full experiment pipeline.
Uses the core framework for all ML logic.

Classes
-------

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