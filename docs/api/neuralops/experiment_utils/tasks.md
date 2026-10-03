Module neuralops.experiment_utils.tasks
=======================================
Background training task (Django integration) with WebSocket broadcasting
and AI assistant.

Functions
---------

`broadcast_terminal(msg: str) ‑> None`
:   Send a log message to the terminal WebSocket group.

`run_training_async(experiment, stop_event: threading.Event) ‑> None`
:   Main entry point called by Django in a background thread.
    
    Args:
        experiment: Django model instance with fields:
            id, config (JSON), status, logs, current_step, total_steps, etc.
        stop_event: threading.Event to signal cancellation.

`save_artifacts(experiment, final_metrics: dict) ‑> None`
:   Save model, plots, history, etc. to the experiment's media directory.

Classes
-------

`AIAssistant(llm_function)`
:   Monitors training metrics and suggests actions (e.g., early stop).
    Uses a callable LLM function for decision making.

    ### Methods

    `monitor(self, exp_id: str, metrics_history: list) ‑> None`
    :   Monitor.

`WSMessageQueue()`
:   Thread‑safe WebSocket message sender.

    ### Static methods

    `send(experiment_id: str, data: dict) ‑> None`
    :   Send.