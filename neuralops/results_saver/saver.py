"""Core ResultsSaver orchestrator class."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
import tarfile
from typing import Any

from neuralops.results_saver.config import SaveConfig
from neuralops.results_saver.storage.sqlite_ops import (
    configure_sqlite_pragmas,
    sanitize_json_value,
    sqlite_file_lock,
)
from neuralops.results_saver.writers.csv_writer import MASTER_CSV_COLUMNS, save_result
from neuralops.results_saver.writers.json_parquet_writer import (
    save_result_json,
    save_trajectories_parquet,
)
from neuralops.results_saver.writers.plot_writer import generate_plots_for_run

logger = logging.getLogger(__name__)


class ResultsSaver:
    """Central orchestrator for experiment result persistence and cloud tracking.

    Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
        saver.push_all()
    """

    def __init__(self, run_id: str, base_dir: Path, config: dict[str, Any] | None = None) -> None:
        """Initialize ResultsSaver directory layouts and configuration.

        Args:
            run_id: Unique identifier string for this run.
            base_dir: Base directory where run directory is created.
            config: Optional configuration dictionary.
        """
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.cfg = SaveConfig.from_dict(config or {})
        self._init_dirs()
        self.saved_paths: dict[str, str] = {}

    def _init_dirs(self) -> None:
        """Initialize required result directory structures."""
        dirs = [self.run_dir]
        if self.cfg.save_logs:
            dirs.append(self.run_dir / "logs")
        if self.cfg.save_plots:
            dirs.append(self.run_dir / "plots")
        if self.cfg.save_artifacts or self.cfg.save_model or self.cfg.model_export_formats:
            dirs.append(self.run_dir / "artifacts")
        if self.cfg.save_checkpoints:
            dirs.append(self.run_dir / "checkpoints")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def should_save_heavy(self, final_metrics: dict[str, Any]) -> bool:
        """Evaluate if metric meets auto_save_if_better_than threshold.

        Args:
            final_metrics: Mapping of evaluation metric names to values.

        Returns:
            True if heavy model artifacts should be persisted.
        """
        threshold = self.cfg.auto_save_if_better_than
        if threshold is None:
            return True
        metric = final_metrics.get("eval_accuracy", 0)
        return metric >= threshold

    def _get_sqlite_connection(self, db_path: Path) -> sqlite3.Connection:
        """Open and configure an SQLite database connection with WAL mode and pragmas."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        configure_sqlite_pragmas(conn)
        return conn

    def save_to_sqlite(
        self,
        db_path: str | Path | None = None,
        config: Any | None = None,
        metrics: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save run metadata, config, and metrics into SQLite with multi-GPU write safety.

        Args:
            db_path: Target SQLite database file path.
            config: Configuration dictionary or object.
            metrics: Optional final metrics dictionary.
            extra_metadata: Optional extra metadata dictionary.

        Returns:
            String path to the SQLite database.
        """
        target_db = Path(db_path) if db_path is not None else (self.base_dir / "experiments.db")
        lock_path = target_db.with_suffix(".db.lock")

        cfg_dict = self._flatten_config(config) if config is not None else {}
        metrics_dict = sanitize_json_value(metrics or {})
        meta_dict = sanitize_json_value(extra_metadata or {})

        with sqlite_file_lock(lock_path):
            conn = self._get_sqlite_connection(target_db)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS experiment_runs (
                        run_id TEXT PRIMARY KEY,
                        created_at TEXT,
                        task_name TEXT,
                        model_name TEXT,
                        config_json TEXT,
                        metrics_json TEXT,
                        metadata_json TEXT
                    )
                    """)
                now_str = datetime.now().isoformat()
                task_name = str(cfg_dict.get("task_name", "unknown"))
                model_name = str(cfg_dict.get("model_name", "unknown"))

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO experiment_runs
                    (run_id, created_at, task_name, model_name, config_json, metrics_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_id,
                        now_str,
                        task_name,
                        model_name,
                        json.dumps(sanitize_json_value(cfg_dict), allow_nan=False),
                        json.dumps(metrics_dict, allow_nan=False),
                        json.dumps(meta_dict, allow_nan=False),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        self.saved_paths["sqlite_db"] = str(target_db)
        return str(target_db)

    def log_metric_to_sqlite(
        self,
        step: int,
        metric_name: str,
        value: float,
        db_path: str | Path | None = None,
    ) -> None:
        """Append a single step metric into SQLite with multi-GPU concurrency safety.

        Args:
            step: Training or evaluation step index.
            metric_name: Metric name identifier.
            value: Numeric metric value.
            db_path: Optional explicit SQLite path.
        """
        target_db = Path(db_path) if db_path is not None else (self.run_dir / "metrics.db")
        lock_path = target_db.with_suffix(".db.lock")

        with sqlite_file_lock(lock_path):
            conn = self._get_sqlite_connection(target_db)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metric_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT,
                        step INTEGER,
                        metric_name TEXT,
                        value REAL,
                        timestamp TEXT
                    )
                    """)
                cursor.execute(
                    """
                    INSERT INTO metric_logs (run_id, step, metric_name, value, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_id,
                        step,
                        metric_name,
                        float(value),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def save_metrics(self, metrics: dict[str, Any]) -> str:
        """Save a dictionary of evaluation/training metrics to JSON and SQLite.

        Args:
            metrics: Metrics mapping.

        Returns:
            Path to final_metrics.json.
        """
        self._save_final_metrics(metrics)
        self.save_to_sqlite(metrics=metrics)
        path = str(self.run_dir / "artifacts" / "final_metrics.json")
        self.saved_paths["metrics"] = path
        return path

    def save_config(self, config: Any) -> str:
        """Save experiment configuration dictionary or object.

        Args:
            config: Config dict or object.

        Returns:
            Path to config.json.
        """
        self._save_config(config)
        self.save_to_sqlite(config=config)
        path = str(self.run_dir / "artifacts" / "config.json")
        self.saved_paths["config"] = path
        return path

    def save_model_weights(self, model: Any, filename: str = "model.pt") -> str:
        """Save PyTorch model state dict or checkpoint.

        Args:
            model: PyTorch model instance.
            filename: Target file name in artifacts directory.

        Returns:
            Path to saved model weights.
        """
        import torch

        art_dir = self.run_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        dest = art_dir / filename
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), dest)
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(str(art_dir / "model"))
            return str(art_dir / "model")
        else:
            torch.save(model, dest)
        self.saved_paths["model_weights"] = str(dest)
        return str(dest)

    def save_result(
        self,
        result: dict[str, Any] | Any,
        results_dir: str | Path | None = None,
        csv_filename: str = "master_results.csv",
    ) -> str:
        """Append a benchmark result to master CSV.

        Args:
            result: Result dictionary or object.
            results_dir: Optional custom results dir.
            csv_filename: Master CSV filename.

        Returns:
            Path to master CSV.
        """
        target_dir = results_dir if results_dir is not None else self.base_dir
        path = save_result(result, results_dir=target_dir, csv_filename=csv_filename)
        self.saved_paths["master_csv"] = path
        return path

    def save_all(
        self,
        model: Any,
        config: Any,
        history: Any,
        final_metrics: dict[str, Any],
        tokenizer: Any = None,
        input_example: Any = None,
    ) -> None:
        """Save everything configured: metrics, config, model, plots, and trackers.

        Args:
            model: Trained model.
            config: Run configuration.
            history: Training history log.
            final_metrics: Summary evaluation metrics.
            tokenizer: Optional tokenizer to save.
            input_example: Optional sample input for export conversions.
        """
        self._save_config(config)
        self._save_history(history)
        self._save_final_metrics(final_metrics)
        self._save_metrics_log(history)
        self.save_to_sqlite(config=config, metrics=final_metrics)
        if self.cfg.save_model and self.should_save_heavy(final_metrics):
            self._save_model(model, tokenizer)
        if self.cfg.save_checkpoints:
            self._save_checkpoint(model, "final")
        if self.cfg.save_plots:
            self._generate_plots(history, final_metrics)
        if self.cfg.save_model and self.should_save_heavy(final_metrics):
            for fmt in self.cfg.model_export_formats:
                self._export_format(fmt, model, input_example)
        self._save_archive()
        self._save_model_card(config, final_metrics)

        if self.cfg.experiment_trackers:
            try:
                from neuralops.tracking import ExperimentTracker

                tracker = ExperimentTracker(
                    backend=self.cfg.experiment_trackers[0],
                    log_dir=str(self.run_dir / "tracking"),
                    experiment_name=self.run_id,
                )
                for k, v in final_metrics.items():
                    tracker.log_metric(k, v)
                tracker.log_params(self._flatten_config(config))
                tracker.finalize()
            except ImportError:
                self.track_all(final_metrics, self._flatten_config(config))

    def _write_json(self, rel_path: str, obj: Any) -> None:
        """Serialize an object to JSON at run_dir / rel_path."""

        def _default_serializer(o: Any) -> Any:
            if hasattr(o, "to_dict") and callable(o.to_dict):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            if isinstance(o, (datetime, Path)):
                return str(o)
            if hasattr(o, "value"):
                return o.value
            return str(o)

        fp = self.run_dir / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(
                sanitize_json_value(obj),
                f,
                indent=2,
                default=_default_serializer,
                allow_nan=False,
            )

    def _write_text(self, rel_path: str, text: str) -> None:
        """Write plaintext content to run_dir / rel_path."""
        fp = self.run_dir / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text)

    def _flatten_config(self, config: Any) -> dict[str, Any]:
        """Convert a configuration object to a flat dictionary."""
        if hasattr(config, "__dict__"):
            return vars(config)
        return config if isinstance(config, dict) else {"config": str(config)}

    def _save_config(self, config: Any) -> None:
        self._write_json("artifacts/config.json", self._flatten_config(config))

    def _save_history(self, history: Any) -> None:
        self._write_json("artifacts/history.json", history)

    def _save_final_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json("artifacts/final_metrics.json", metrics)

    def _save_metrics_log(self, history: Any) -> None:
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "metrics.jsonl", "w", encoding="utf-8") as f:
            for entry in history:
                f.write(json.dumps(sanitize_json_value(entry), allow_nan=False) + "\n")

    def _save_model(self, model: Any, tokenizer: Any = None) -> None:
        p = self.run_dir / "artifacts" / "model"
        p.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(p))
        if tokenizer and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(p))
        if self.cfg.save_optimizer and hasattr(model, "state_dict"):
            import torch

            torch.save(model.state_dict(), p / "optimizer.pt")

    def _save_checkpoint(self, model: Any, name: str = "best") -> None:
        p = self.run_dir / "checkpoints" / name
        p.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(p))

    def _save_archive(self) -> None:
        try:
            with tarfile.open(self.run_dir / "results.tar.gz", "w:gz") as tar:
                for sub in ["artifacts", "logs", "plots"]:
                    sp = self.run_dir / sub
                    if sp.exists():
                        tar.add(sp, arcname=sub)
        except Exception as e:
            logger.warning(f"Archive creation skipped due to warning/error: {e}")

    def _generate_plots(self, history: Any, final_metrics: dict[str, Any]) -> None:
        generate_plots_for_run(self.run_dir, self.cfg, history, final_metrics)

    def _save_model_card(self, config: Any, metrics: dict[str, Any]) -> None:
        cfg_dict = self._flatten_config(config)
        model_name = cfg_dict.get("model_name", "Model")
        task_name = cfg_dict.get("task_name", "Task")
        acc = metrics.get("eval_accuracy", "N/A")
        loss = metrics.get("eval_loss", "N/A")
        card = f"""# {model_name} – {task_name}

**Accuracy:** {acc}
**Loss:** {loss}

**Run ID:** {self.run_id}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Compression:** {cfg_dict.get("attention_type", "full")} (rank={cfg_dict.get("rank", "N/A")})
"""
        self._write_text("artifacts/model_card.md", card)

    def _export_format(self, fmt: str, model: Any, input_example: Any) -> None:
        try:
            from neuralops.results_saver.backends.exports import EXPORTERS

            exporter = EXPORTERS.get(fmt)
            if exporter:
                exporter(model, input_example, str(self.run_dir / "artifacts"))
        except Exception:
            logger.exception(f"Export {fmt} failed")

    def push_all(self) -> None:
        """Push results to all configured storage backends."""
        for be in self.cfg.storage_backends:
            self._push_backend(be)

    def _push_backend(self, name: str) -> None:
        try:
            if name == "local":
                return
            mod = __import__(f"neuralops.results_saver.backends.{name}", fromlist=["push"])
            mod.push(self.run_dir, self.cfg)
        except ImportError:
            logger.warning(f"Backend '{name}' not available, skipping")
        except Exception:
            logger.exception(f"Push to {name} failed")

    def git_commit_and_push(self) -> None:
        """Trigger git commit and push if enabled in git_options."""
        if not self.cfg.git_options.get("commit_results"):
            return
        from neuralops.results_saver.backends.git import git_commit_and_push

        git_commit_and_push(self.run_dir, self.cfg)

    def dvc_track(self) -> None:
        """Track result directory with DVC if enabled."""
        if not self.cfg.dvc_options.get("track_model"):
            return
        from neuralops.results_saver.backends.dvc import dvc_track

        dvc_track(self.run_dir, self.cfg)

    def track_all(self, metrics: dict[str, Any], params: dict[str, Any]) -> None:
        """Dispatch metrics and params to configured experiment trackers."""
        for t in self.cfg.experiment_trackers:
            try:
                mod = __import__(
                    f"neuralops.results_saver.backends.{t}_tracker", fromlist=["track"]
                )
                mod.track(self.run_id, self.run_dir, metrics, params, self.cfg)
            except ImportError:
                logger.warning(f"Tracker '{t}' not available, skipping")
            except Exception:
                logger.exception(f"Tracking to {t} failed")

    def push_to_registries(self) -> None:
        """Upload model checkpoints to configured registries."""
        for r in self.cfg.model_registries:
            try:
                mod = __import__(f"neuralops.results_saver.backends.{r}", fromlist=["push"])
                mod.push(self.run_id, self.run_dir, self.cfg)
            except ImportError:
                logger.warning(f"Registry '{r}' not available")
            except Exception:
                logger.exception(f"Push to registry {r} failed")

    def create_github_release(self, metrics: dict[str, Any]) -> str | None:
        """Create a GitHub release if enabled in release options."""
        if not self.cfg.release_options.get("create_github_release"):
            return None
        from neuralops.results_saver.backends.github_release import create_release

        return create_release(self.run_id, self.run_dir, metrics, self.cfg)

    def notify(self, success: bool, metrics: dict[str, Any]) -> None:
        """Send run completion notifications."""
        from neuralops.results_saver.backends.notification import notify

        notify(self.run_id, success, metrics, self.cfg)
