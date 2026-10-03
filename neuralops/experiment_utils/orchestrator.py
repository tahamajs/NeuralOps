"""
MegaOrchestrator – single entry point that executes the full experiment pipeline.
Uses the core framework for all ML logic.
"""

import logging
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neuralops.config import Config, load_config
from neuralops.data import DataConfig, TaskType, load_dataset, preprocess_function
from neuralops.models import load_model
from neuralops.results_saver import ResultsSaver, SaveConfig
from neuralops.training import HuggingFaceTrainer, ResultsManager, get_task_metrics

logger = logging.getLogger(__name__)


class MegaOrchestrator:
    """
    Central orchestrator that connects all framework modules.

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
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_id: str,
        base_dir: Path,
        progress_callback: Callable | None = None,
        log_callback: Callable | None = None,
        final_callback: Callable | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.raw_config = config
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.progress_cb = progress_callback
        self.log_cb = log_callback
        self.final_cb = final_callback
        self.stop_event = stop_event or threading.Event()

        # Convert dict to unified Config
        self.cfg = load_config(None)  # start with defaults
        if config:
            self.cfg = Config(
                **{k: v for k, v in config.items() if k in Config.__dataclass_fields__}
            )

        self.model = None
        self.tokenizer = None
        self.trainer_state = None
        self.final_metrics: dict[str, float] = {}

        logger.info("MegaOrchestrator created: run_id=%s", run_id)

    def run(self) -> dict[str, Any]:
        """Execute the full experiment pipeline."""
        try:
            self._log("Starting experiment…")

            # 1. Load model & tokenizer
            self._log("Loading model…")
            self.model, self.tokenizer = load_model(
                self.cfg.model.name,
                num_labels=self.cfg.model.num_labels,
                attention_type=self.cfg.compression.attn_type,
                attention_rank=self.cfg.compression.rank,
            )

            # 2. Apply training method (if any)
            if self.cfg.method.method != "baseline":
                from neuralops.trainer.methods import get_method

                method_fn = get_method(self.cfg.method.method)
                if method_fn:
                    method_fn(self.model, self.cfg.compression)

            # 3. Load data
            self._log("Loading data…")
            exp_cfg = self._build_experiment_config()
            dataset = load_dataset(exp_cfg)
            tokenized = self._preprocess_dataset(dataset)

            # 4. Build trainer
            self._log("Building trainer…")
            results = ResultsManager(self.run_id, self.base_dir, config=self.raw_config)
            trainer = HuggingFaceTrainer(
                model=self.model,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized.get("validation"),
                config=self.cfg.training.__dict__,
                results_manager=results,
                compute_metrics=get_task_metrics(self.cfg.training.task),
            )
            # Add basic callbacks
            from neuralops.trainer.callbacks import MemoryCallback, TimeCallback

            trainer.add_callback(MemoryCallback())
            trainer.add_callback(TimeCallback())

            # 5. Train
            self._log("Training started…")
            trainer.train()
            eval_metrics = trainer.evaluate()
            self.final_metrics = eval_metrics
            self.trainer_state = (
                trainer.state.log_history if hasattr(trainer, "state") else []
            )

            # 6. Save everything
            self._log("Saving results…")
            saver = ResultsSaver(
                run_id=self.run_id,
                base_dir=self.base_dir,
                config=SaveConfig(save_model=True, save_plots=True),
            )
            saver.save_all(
                self.model,
                self.cfg.__dict__,
                self.trainer_state,
                eval_metrics,
                self.tokenizer,
            )
            saver.push_all()

            self._log("Experiment completed.")
            return eval_metrics

        except Exception as e:
            logger.exception("Orchestrator failed: %s", e)
            self._log(f"Experiment failed: {traceback.format_exc()}")
            raise

    def _build_experiment_config(self) -> Any:
        """build experiment config."""
        from neuralops.data.dataset_loader import ExperimentConfig

        return ExperimentConfig(dataset_name=self.cfg.training.task)

    def _preprocess_dataset(self, dataset) -> Any:
        """preprocess dataset."""
        task = self.cfg.training.task
        s1, s2 = None, None
        from neuralops.data.glue_loader import GLUE_TASK_KEYS

        if task in GLUE_TASK_KEYS:
            s1, s2 = GLUE_TASK_KEYS[task]
        data_cfg = DataConfig(
            task_type=(
                TaskType.SINGLE_CLASSIFICATION
                if s2 is None
                else TaskType.PAIR_CLASSIFICATION
            ),
            max_length=self.cfg.data.max_seq_length,
            batch_size=self.cfg.training.batch_size,
            text_column_1=s1,
            text_column_2=s2,
            label_column="label",
        )

        def preprocess(examples) -> Any:
            """Preprocess."""
            return preprocess_function(examples, self.tokenizer, data_cfg)

        tokenized = dataset.map(
            preprocess, batched=True, remove_columns=dataset["train"].column_names
        )
        tokenized.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return tokenized

    def _log(self, msg: str) -> None:
        """log."""
        logger.info("[%s] %s", self.run_id[:8], msg)
        if self.log_cb:
            self.log_cb(msg)
