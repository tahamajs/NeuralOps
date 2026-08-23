"""
MegaOrchestrator – single entry point for the entire platform.
Receives one config dictionary and executes the full pipeline.
"""

import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        config: Dict[str, Any],
        run_id: str,
        base_dir: Path,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        final_callback: Optional[Callable] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / str(run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.progress_cb = progress_callback
        self.log_cb = log_callback
        self.final_cb = final_callback
        self.stop_event = stop_event or threading.Event()

        # These will be populated during run()
        self.model = None
        self.tokenizer = None
        self.trainer_state = None
        self.final_metrics: Dict[str, float] = {}

        # Log initial config
        logger.info("MegaOrchestrator created: run_id=%s", run_id)

    # ════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ════════════════════════════════════════════════════════════════
    def run(self) -> Dict[str, Any]:
        """Execute the full experiment pipeline."""
        try:
            # Phase 0 – save config
            self._save_config()

            # Phase 1 – Hyperparameter Optimization (optional)
            if self.config.get("hpo", {}).get("enabled"):
                self._log("Starting HPO…")
                self.config = self._run_hpo()
                self._log(
                    f"HPO finished. Best params: {self.config.get('hpo_best_params', {})}"
                )

            # Phase 2 – Federated Learning (optional)
            if self.config.get("federated_learning", {}).get("enabled"):
                self._log("Starting federated learning…")
                self.final_metrics = self._run_federated()
                return self.final_metrics

            # Phase 3 – Single training run
            self._log("Starting training…")
            self.final_metrics = self._run_single_training()
            self._log(f"Training finished. Metrics: {self.final_metrics}")

            # Phase 4 – Post‑training analysis
            self._run_post_analysis()

            # Phase 5 – Final callback
            if self.final_cb:
                self.final_cb(self.final_metrics, self.trainer_state)

            self._log("Experiment completed successfully.")
            return self.final_metrics

        except Exception as e:
            logger.exception("Orchestrator failed: %s", e)
            self._log(f"Experiment failed: {traceback.format_exc()}")
            if self.final_cb:
                self.final_cb({"error": str(e)}, None)
            raise

    # ════════════════════════════════════════════════════════════════
    # Phase 1 – HPO
    # ════════════════════════════════════════════════════════════════
    def _run_hpo(self) -> Dict[str, Any]:
        """ run hpo."""
        from framework.hyperopt.integration import run_hpo_experiment

        hpo_cfg = self.config.get("hpo", {})

        def trial_objective(trial_params: Dict[str, Any]) -> float:
            """Run one HPO trial. Returns a score to maximize."""
            merged = {**self.config, **trial_params}
            merged["hpo"] = {"enabled": False}  # prevent recursion
            trial_orch = MegaOrchestrator(
                config=merged,
                run_id=f"{self.run_id}-hpo-{datetime.now().strftime('%H%M%S')}",
                base_dir=self.base_dir,
            )
            metrics = trial_orch._run_single_training()
            return metrics.get("eval_accuracy", 0)

        result = run_hpo_experiment(
            hpo_cfg, trial_objective, output_dir=self.run_dir / "hpo"
        )
        self.config["hpo_best_params"] = result["best_params"]
        self.config.update(result["best_params"])
        return self.config

    # ════════════════════════════════════════════════════════════════
    # Phase 2 – Federated Learning
    # ════════════════════════════════════════════════════════════════
    def _run_federated(self) -> Dict[str, Any]:
        """ run federated."""
        from framework.federated_learning.integration import run_federated_experiment

        fl_cfg = self.config.get("federated_learning", {})

        def model_builder() -> Any:
            """Model builder."""
            return self._load_model()

        # User must provide a dataset or a dataset loader function
        dataset = self.config.get("_dataset")  # injected by Django view
        result = run_federated_experiment(
            fl_cfg,
            model_builder,
            dataset,
            output_dir=self.run_dir / "fl",
        )
        return result

    # ════════════════════════════════════════════════════════════════
    # Phase 3 – Single Training
    # ════════════════════════════════════════════════════════════════
    def _run_single_training(self) -> Dict[str, float]:
        """ run single training."""
        cfg = self.config

        # 3a. Load model & tokenizer
        self._log("Loading model…")
        self.model, self.tokenizer = self._load_model()

        # 3b. Inject attention
        if cfg.get("attn_type") and cfg.get("attn_type") != "full":
            from framework.models import replace_attention_in_model

            self.model = replace_attention_in_model(self.model, cfg["attn_type"])

        # 3c. Apply training method (LoRA, R-Drop, etc.)
        if cfg.get("method") and cfg.get("method") != "baseline":
            from framework.trainer.methods.registry import METHOD_REGISTRY, get_method

            method_name = cfg["method"]
            method = None
            if hasattr(METHOD_REGISTRY, "resolve"):
                method = METHOD_REGISTRY.resolve(method_name)
            if not method:
                fn = get_method(method_name) if callable(get_method) else METHOD_REGISTRY.get(method_name)
                if fn:
                    class MethodWrapper:
                        def __init__(self, f):
                            self.f = f
                        def apply(self, m, c):
                            res = self.f(m, c)
                            return res if res is not None else m
                    method = MethodWrapper(fn)
            if method:
                self.model = method.apply(self.model, cfg)

        # 3d. Distributed wrapping (optional)
        if cfg.get("distributed", {}).get("strategy", "none") != "none":
            from framework.distributed.integration import wrap_with_distributed

            self.model = wrap_with_distributed(
                cfg.get("distributed", {}), self.model, self.run_id
            )

        # 3e. Build callbacks
        callbacks = self._build_callbacks()

        # 3f. Load data
        train_dataset, eval_dataset, num_labels = self._load_data(cfg)

        # 3g. Create trainer
        trainer = self._create_trainer(
            cfg, train_dataset, eval_dataset, num_labels, callbacks
        )

        # 3h. Train
        self._log("Training started…")
        trainer.train()
        eval_result = trainer.evaluate()

        # 3i. Store state
        self.trainer_state = (
            trainer.state.log_history if hasattr(trainer, "state") else []
        )
        self.final_metrics = eval_result

        # 3j. Save everything
        self._save_results(eval_result)

        # 3k. Model versioning
        self._create_model_version(eval_result)

        # 3l. Cleanup distributed
        if cfg.get("distributed", {}).get("strategy", "none") != "none":
            from framework.distributed.integration import cleanup_distributed

            cleanup_distributed(self.model, self.run_id)

        return eval_result

    # ════════════════════════════════════════════════════════════════
    # Phase 4 – Post‑Training Analysis
    # ════════════════════════════════════════════════════════════════
    def _run_post_analysis(self) -> None:
        """ run post analysis."""
        cfg = self.config

        # Drift detection
        if cfg.get("drift", {}).get("enabled"):
            self._log("Running drift detection…")
            self._run_drift()

        # Fairness
        if cfg.get("fairness", {}).get("enabled"):
            self._log("Evaluating fairness…")
            self._run_fairness()

        # Robustness
        if cfg.get("robustness", {}).get("enabled"):
            self._log("Testing adversarial robustness…")
            self._run_robustness()

        # Notifications
        if cfg.get("save_config", {}).get("notification"):
            from framework.results_saver.backends.notification import notify

            notify(self.run_id, True, self.final_metrics, cfg["save_config"])

    # ── Helpers ─────────────────────────────────────────────────────
    def _load_model(self) -> Any:
        """ load model."""
        from framework.models import load_model

        cfg = self.config
        return load_model(
            model_name=cfg.get("model_name", "bert-base-uncased"),
            num_labels=cfg.get("num_labels", 2),
            loader_type=cfg.get("model_variant", "hf"),
            rank=cfg.get("compression_rank", 64),
        )

    def _load_data(self, cfg) -> Any:
        """ load data."""
        from framework.experiment_utils.config import ExperimentConfig
        from framework.experiment_utils.data import load_and_preprocess_data

        exp_cfg = ExperimentConfig(
            **{
                k: v
                for k, v in cfg.items()
                if k in ExperimentConfig.__dataclass_fields__
            }
        )
        setattr(exp_cfg, "glue_base_path", cfg.get("glue_base_path", "/default/path"))
        return load_and_preprocess_data(exp_cfg)

    def _create_trainer(self, cfg, train_ds, eval_ds, num_labels, callbacks) -> Any:
        """ create trainer."""
        from framework.trainer.trainer_factory import get_trainer
        from transformers import DataCollatorWithPadding, TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(self.run_dir / "checkpoints"),
            num_train_epochs=cfg.get("epochs", 3),
            per_device_train_batch_size=cfg.get("batch_size", 16),
            per_device_eval_batch_size=cfg.get("batch_size", 16) * 2,
            learning_rate=cfg.get("learning_rate", 2e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
            warmup_steps=cfg.get("warmup_steps", 0),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            fp16=(cfg.get("mixed_precision", "no") == "fp16"),
            bf16=(cfg.get("mixed_precision", "no") == "bf16"),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            optim=cfg.get("optimizer_type", "adamw_hf"),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            remove_unused_columns=False,
            report_to="none",
        )
        trainer_cls = get_trainer(cfg.get("trainer_type", "hf"))
        return trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            callbacks=callbacks,
        )

    def _build_callbacks(self) -> Any:
        """ build callbacks."""
        cbs = []
        from Codes.framework.callbacks.django import DjangoCallback

        cbs.append(
            DjangoCallback(
                progress_cb=self.progress_cb,
                log_cb=self.log_cb,
                final_cb=self.final_cb,
                stop_event=self.stop_event,
            )
        )
        # Add more callbacks based on config…
        return cbs

    def _save_results(self, metrics) -> None:
        """ save results."""
        from framework.results_saver import ResultsSaver

        save_cfg = self.config.get("save_config", {})
        saver = ResultsSaver(self.run_id, self.base_dir, save_cfg)
        saver.save_all(
            self.model,
            self.config,
            self.trainer_state,
            metrics,
            tokenizer=self.tokenizer,
        )
        saver.push_all()
        saver.git_commit_and_push()
        saver.track_all(metrics, self.config)

    def _create_model_version(self, metrics) -> None:
        """ create model version."""
        from framework.model_versioning import ModelVersionService, VersioningConfig

        save_cfg = self.config.get("save_config", {})
        vc = VersioningConfig(**save_cfg.get("versioning", {}))
        ModelVersionService.create_version(
            run=None,
            model_state=self.model.state_dict(),
            metrics=metrics,
            config=vc,
            tokenizer=self.tokenizer,
            artifact_base=str(self.base_dir),
        )

    def _run_drift(self) -> None:
        """ run drift."""
        from framework.drift.integration import auto_drift_check

        auto_drift_check(
            self.run_id,
            self.config.get("drift", {}).get("reference", {}),
            self.config.get("drift", {}).get("current", {}),
            output_dir=self.run_dir / "drift",
        )

    def _run_fairness(self) -> None:
        """ run fairness."""
        import json

        import numpy as np
        from framework.evaluation.fairness import FairnessEvaluator

        eval_preds = self.run_dir / "artifacts" / "eval_predictions.json"
        if eval_preds.exists():
            with open(eval_preds) as f:
                data = json.load(f)
            evaluator = FairnessEvaluator()
            report = evaluator.evaluate(
                predictions=np.array(data["predictions"]),
                labels=np.array(data["labels"]),
                groups=data.get("groups", {}),
            )
            path = self.run_dir / "fairness_report.json"
            with open(path, "w") as f:
                json.dump(report.__dict__, f, indent=2, default=str)

    def _run_robustness(self) -> None:
        """ run robustness."""
        import json

        import numpy as np
        from framework.evaluation.robustness import TextAdversarialTester

        eval_preds = self.run_dir / "artifacts" / "eval_predictions.json"
        if eval_preds.exists():
            with open(eval_preds) as f:
                data = json.load(f)
            tester = TextAdversarialTester()
            # Simplified – needs model predict_fn
            report = tester.evaluate(
                predict_fn=lambda texts: np.zeros(len(texts)),
                texts=data.get("texts", []),
                labels=np.array(data.get("labels", [])),
            )
            path = self.run_dir / "robustness_report.json"
            with open(path, "w") as f:
                json.dump(report.__dict__, f, indent=2, default=str)

    def _save_config(self) -> None:
        """ save config."""
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _log(self, msg: str) -> None:
        """ log."""
        logger.info("[%s] %s", self.run_id[:8], msg)
        if self.log_cb:
            self.log_cb(0, {"message": msg})
