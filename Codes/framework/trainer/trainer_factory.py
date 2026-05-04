# framework/training/trainer_factory.py
from typing import Dict, Any, Optional, Tuple
from .results_manager import ResultsManager
from .registry import TrainerRegistry

def get_trainer(
    model,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    results_manager: Optional[ResultsManager] = None,
    trainer_type: str = "hf",          # choose from "hf", "unsloth", "trl", "custom_loop"
    device=None,
    num_labels=None,
    is_multichoice=False,
    compute_metrics=None,
    tokenizer=None,
    **kwargs
) -> Tuple[Any, ResultsManager]:
    """
    Factory to create a trainer using the registry.
    """
    if results_manager is None:
        results_manager = ResultsManager(
            experiment_name=config.get("experiment_name", "experiment"),
            config=config,
        )

    trainer_class = TrainerRegistry.get(trainer_type)
    # Map parameters to the trainer's signature
    trainer = trainer_class(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        results_manager=results_manager,
        device=device,
        num_labels=num_labels,
        is_multichoice=is_multichoice,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        **kwargs
    )
    return trainer, results_manager