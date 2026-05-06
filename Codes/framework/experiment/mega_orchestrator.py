"""
MegaOrchestrator – ties every framework module together.
Called from the Django runner with a single config dict.
"""

import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from django.conf import settings

from framework.experiment.config import ExperimentConfig
from framework.models import load_model, replace_attention_in_model
from framework.trainer.trainer_factory import get_trainer
from framework.logging.integration import auto_setup as logging_setup
from framework.callbacks.django_integration import DjangoCallback
from framework.results_saver import ResultsSaver, SaveConfig
from framework.model_versioning import ModelVersionService, VersioningConfig
from framework.drift.integration import auto_drift_check
from framework.evaluation.fairness import FairnessEvaluator
from framework.evaluation.robustness import TextAdversarialTester
from framework.hyperopt.integration import run_hpo_experiment
from framework.federated_learning.integration import run_federated_experiment
from framework.analysis.comparison import ComparisonEngine

logger = logging.getLogger(__name__)

class MegaOrchestrator:
    """
    One‑stop orchestrator for any experiment type.
    Reads the full config blob from the Django API and executes everything.
    """

    def __init__(self, config: Dict[str, Any], run_id: str,
                 progress_cb=None, log_cb=None, final_cb=None, stop_event=None):
        self.config = config
        self.run_id = run_id
        self.base_dir = Path(settings.MEDIA_ROOT) / run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks for Django dashboard
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.final_cb = final_cb
        self.stop_event = stop_event or threading.Event()

        # Setup logging & tracking
        self.log_components = logging_setup(config, run_id, Path(settings.MEDIA_ROOT))

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and return final metrics."""
        logger.info("MegaOrchestrator started for run %s", self.run_id)

        # --- 1. Federated Learning (optional) ---
        fl_config = self.config.get("federated_learning", {})
        if fl_config.get("enabled"):
            return self._run_federated(fl_config)

        # --- 2. Hyperparameter Optimization (optional) ---
        hpo_config = self.config.get("hpo", {})
        if hpo_config.get("enabled"):
            best_params = run_hpo_experiment(hpo_config, self._run_single_trial,
                                             output_dir=self.base_dir / "hpo")
            self.config.update(best_params["best_params"])
            logger.info("HPO finished: best params = %s", best_params["best_params"])

        # --- 3. Single training run ---
        final_metrics = self._run_single_trial(self.config)

        # --- 4. Post‑training analysis ---
        self._post_training_analysis(final_metrics)

        # --- 5. Final callback ---
        if self.final_cb:
            self.final_cb(final_metrics, {})
        logger.info("MegaOrchestrator finished for run %s", self.run_id)
        return final_metrics

    # ------------------------------------------------------------------
    def _run_single_trial(self, trial_config: Dict[str, Any]) -> Dict[str, Any]:
        """Core training loop (used by HPO too)."""
        cfg = self._build_experiment_config(trial_config)

        # 1. Load model
        model, tokenizer = load_model(
            model_name=cfg.model_name,
            num_labels=getattr(cfg, 'num_labels', 2),
            loader_type=getattr(cfg, 'model_variant', 'hf'),
            rank=getattr(cfg, 'compression_rank', 64),
        )

        # 2. Inject attention
        if cfg.attn_type and cfg.attn_type != "full":
            model = replace_attention_in_model(model, cfg.attn_type)

        # 3. Build callbacks
        callbacks = [DjangoCallback(self.progress_cb, self.log_cb, self.final_cb, self.stop_event)]
        # (MemoryCallback, TimeCallback, etc. are added inside the trainer factory)

        # 4. Create trainer
        trainer = get_trainer(getattr(cfg, 'trainer_type', 'hf'))(
            model=model, dataset=self._load_data(cfg), config=cfg,
            tokenizer=tokenizer, callbacks=callbacks,
        )

        # 5. Train
        trainer.train()
        eval_result = trainer.evaluate()

        # 6. Save results (optional – controlled by save_config)
        save_cfg = self.config.get("save_config", {})
        saver = ResultsSaver(self.run_id, Path(settings.MEDIA_ROOT), save_cfg)
        saver.save_all(model, trial_config, trainer.state.log_history, eval_result,
                       tokenizer=tokenizer)

        # 7. Versioning
        version_cfg = VersioningConfig(**save_cfg.get("versioning", {}))
        ModelVersionService.create_version(
            run=None,   # in production, link to your Django TrainingRun
            model_state=model.state_dict(),
            metrics=eval_result,
            config=version_cfg,
            tokenizer=tokenizer,
        )

        return eval_result

    # ------------------------------------------------------------------
    def _run_federated(self, fl_config: Dict[str, Any]) -> Dict[str, Any]:
        """Federated learning path."""
        def model_builder():
            from framework.models import load_model
            m, _ = load_model(fl_config["model_name"], num_labels=fl_config.get("num_labels",2))
            return m
        dataset = ...   # user provides dataset loader
        return run_federated_experiment(fl_config, model_builder, dataset,
                                        output_dir=self.base_dir / "fl")

    # ------------------------------------------------------------------
    def _post_training_analysis(self, final_metrics: Dict[str, float]) -> None:
        """Optional drift detection, fairness, robustness."""
        if self.config.get("drift", {}).get("enabled"):
            # compare training distribution with a reference
            auto_drift_check(self.run_id, {}, {}, output_dir=self.base_dir / "drift")

        if self.config.get("fairness", {}).get("enabled"):
            eval = FairnessEvaluator()
            report = eval.evaluate(...)   # needs predictions + group arrays
            report.save(self.base_dir / "fairness_report.json")

        if self.config.get("robustness", {}).get("enabled"):
            tester = TextAdversarialTester()
            report = tester.evaluate(...)
            report.save(self.base_dir / "robustness_report.json")

    # ------------------------------------------------------------------
    def _build_experiment_config(self, d: dict):
        """Convert raw dict to ExperimentConfig."""
        from framework.experiment.config import ExperimentConfig
        valid = {k: v for k, v in d.items() if k in ExperimentConfig.__dataclass_fields__}
        return ExperimentConfig(**valid)

    def _load_data(self, cfg):
        """Placeholder – replace with your actual DataLoader setup."""
        return ...   # your existing data loading code