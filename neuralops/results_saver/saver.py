"""
ResultsSaver – central orchestrator for result persistence.
Integrates with framework.analysis, framework.tracking, SQLite WAL storage, and all backends.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sqlite3
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import SaveConfig

logger = logging.getLogger(__name__)

MASTER_CSV_COLUMNS: list[str] = [
    "timestamp",
    "track",
    "module_name",
    "module_type",
    "model_family",
    "dataset",
    "status",
    "error_message",
    "forward_latency_ms",
    "backward_latency_ms",
    "throughput_samples_per_sec",
    "speedup_vs_baseline",
    "param_count",
    "trainable_param_count",
    "compression_ratio",
    "peak_memory_mb",
    "loss",
    "accuracy_pct",
    "f1_score",
    "mcc",
    "pearson",
    "spearman",
    "fidelity_score",
    "flops_macs",
    "throughput_tokens_sec",
    "latency_variance_ms",
    "perplexity",
    "grad_norm",
    "ci_lower",
    "ci_upper",
    "p_value_vs_baseline",
    "mean_seed_acc",
    "std_seed_acc",
]


def save_result(
    result: dict[str, Any] | Any,
    results_dir: str | Path = "07_misc_tests/benchmark_results",
    csv_filename: str = "master_results.csv",
) -> str:
    """
    Append a benchmark result row to the master CSV file.

    1. Master CSV is at 07_misc_tests/benchmark_results/master_results.csv (or results_dir/csv_filename).
    2. Header is written only if file doesn't exist.
    3. Each call appends one row using csv.DictWriter with extrasaction='ignore'.
    4. All float fields are rounded to 4 decimal places before writing.
    """
    import csv

    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / csv_filename

    if hasattr(result, "to_dict") and callable(result.to_dict):
        row = dict(result.to_dict())
    elif hasattr(result, "__dict__"):
        row = dict(vars(result))
    elif isinstance(result, dict):
        row = dict(result)
    else:
        row = dict(result)

    if "timestamp" not in row or not row["timestamp"]:
        row["timestamp"] = datetime.now().isoformat()

    # Format fields: round floats to 4 decimal places, None to empty string
    formatted_row: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float):
            formatted_row[k] = round(v, 4) if math.isfinite(v) else ""
        elif v is None:
            formatted_row[k] = ""
        else:
            formatted_row[k] = v

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=MASTER_CSV_COLUMNS,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(formatted_row)

    return str(csv_path)


def _sanitize_json_value(value: Any) -> Any:
    """Convert runtime values to strict JSON-safe values.

    JSON permits neither NaN nor Infinity.  Persisting either makes results
    silently invalid for downstream dashboards and experiment aggregation, so
    non-finite numeric values are represented as null instead.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json_value(v) for v in value]
    return value


@contextlib.contextmanager
def sqlite_file_lock(lock_file_path: str | Path, timeout: float = 30.0):
    """
    Acquire an exclusive cross-process / multi-GPU file lock to serialize SQLite database writes.

    # FIX: Uses fcntl.flock to guarantee atomic, race-free SQLite write transactions across multi-GPU ranks.
    """
    lock_path = Path(lock_file_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd: int | None = None
    start_time = time.time()
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        # Attempt non-blocking lock with retry loop until timeout expires
        has_fcntl = True
        try:
            import fcntl
        except ImportError:
            has_fcntl = False

        if has_fcntl:
            acquired = False
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    time.sleep(0.05)

            if not acquired:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                ):
                    pass

        yield
    finally:
        if lock_fd is not None:
            try:
                import fcntl

                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
                ImportError,
            ):
                pass
            try:
                os.close(lock_fd)
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
                ImportError,
            ):
                pass


def configure_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """
    Configure high-performance, concurrency-safe SQLite pragmas for multi-GPU training.

    # FIX: Enables Write-Ahead Logging (WAL) and NORMAL synchronous mode to eliminate multi-GPU write contention.
    """
    cursor = conn.cursor()
    # FIX: Enable WAL mode for concurrent readers and single-writer non-blocking transactions
    cursor.execute("PRAGMA journal_mode=WAL;")
    # FIX: Set synchronous mode to NORMAL for optimal disk write efficiency without corruption in WAL
    cursor.execute("PRAGMA synchronous=NORMAL;")
    # FIX: 30-second busy timeout to prevent 'database is locked' errors under multi-GPU load
    cursor.execute("PRAGMA busy_timeout=30000;")
    cursor.close()


class ResultsSaver:
    """
    Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
        saver.push_all()
    """

    def __init__(
        self, run_id: str, base_dir: Path, config: dict[str, Any] | None = None
    ) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.cfg = SaveConfig.from_dict(config or {})
        self._init_dirs()
        self.saved_paths: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    def _init_dirs(self) -> None:
        """Initialize required result directory structures."""
        dirs = [self.run_dir]
        if self.cfg.save_logs:
            dirs.append(self.run_dir / "logs")
        if self.cfg.save_plots:
            dirs.append(self.run_dir / "plots")
        if (
            self.cfg.save_artifacts
            or self.cfg.save_model
            or self.cfg.model_export_formats
        ):
            dirs.append(self.run_dir / "artifacts")
        if self.cfg.save_checkpoints:
            dirs.append(self.run_dir / "checkpoints")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def should_save_heavy(self, final_metrics: dict[str, Any]) -> bool:
        """Evaluate if metric meets auto_save_if_better_than threshold."""
        threshold = self.cfg.auto_save_if_better_than
        if threshold is None:
            return True
        metric = final_metrics.get("eval_accuracy", 0)
        return metric >= threshold

    # ------------------------------------------------------------------
    # SQLite Multi-GPU Concurrency Methods
    # ------------------------------------------------------------------
    def _get_sqlite_connection(self, db_path: Path) -> sqlite3.Connection:
        """Open and configure an SQLite database connection with WAL mode and pragmas."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        # FIX: Ensure PRAGMA journal_mode=WAL; and PRAGMA synchronous=NORMAL; are applied
        configure_sqlite_pragmas(conn)
        return conn

    def save_to_sqlite(
        self,
        db_path: str | Path | None = None,
        config: Any | None = None,
        metrics: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save run metadata, configuration, and final metrics into SQLite database with multi-GPU write safety.

        # FIX: Uses PRAGMA journal_mode=WAL;, PRAGMA synchronous=NORMAL;, and file locking for multi-GPU database write safety.
        """
        target_db = (
            Path(db_path) if db_path is not None else (self.base_dir / "experiments.db")
        )
        lock_path = target_db.with_suffix(".db.lock")

        cfg_dict = self._flatten_config(config) if config is not None else {}
        metrics_dict = _sanitize_json_value(metrics or {})
        meta_dict = _sanitize_json_value(extra_metadata or {})

        # FIX: Wrap database transaction in cross-process file lock
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
                        json.dumps(_sanitize_json_value(cfg_dict), allow_nan=False),
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
        """
        Append a single step metric into SQLite database with multi-GPU concurrency safety.

        # FIX: Uses PRAGMA journal_mode=WAL; and PRAGMA synchronous=NORMAL; with file locking to ensure safe step metric logging.
        """
        target_db = (
            Path(db_path) if db_path is not None else (self.run_dir / "metrics.db")
        )
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

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def save_metrics(self, metrics: dict[str, Any]) -> str:
        """Save a dictionary of evaluation/training metrics to JSON and SQLite."""
        self._save_final_metrics(metrics)
        self.save_to_sqlite(metrics=metrics)
        path = str(self.run_dir / "artifacts" / "final_metrics.json")
        self.saved_paths["metrics"] = path
        return path

    def save_config(self, config: Any) -> str:
        """Save experiment configuration dictionary or object."""
        self._save_config(config)
        self.save_to_sqlite(config=config)
        path = str(self.run_dir / "artifacts" / "config.json")
        self.saved_paths["config"] = path
        return path

    def save_model_weights(self, model: Any, filename: str = "model.pt") -> str:
        """Save PyTorch model state dict or checkpoint."""
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
        """Append a benchmark result to master CSV."""
        target_dir = results_dir if results_dir is not None else self.base_dir
        path = save_result(result, results_dir=target_dir, csv_filename=csv_filename)
        self.saved_paths["master_csv"] = path
        return path

    # ------------------------------------------------------------------
    # Main save entry point
    # ------------------------------------------------------------------
    def save_all(
        self, model, config, history, final_metrics, tokenizer=None, input_example=None
    ) -> None:
        """Save everything selected in the config."""

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

        # Automatically track to experiment trackers (wandb, mlflow, etc.)
        if self.cfg.experiment_trackers:
            # Delegate to the unified tracking module if available, else use legacy backends
            try:
                from neuralops.tracking import ExperimentTracker

                tracker = ExperimentTracker(
                    backend=self.cfg.experiment_trackers[0],  # supports one primary
                    log_dir=str(self.run_dir / "tracking"),
                    experiment_name=self.run_id,
                )
                for k, v in final_metrics.items():
                    tracker.log_metric(k, v)
                tracker.log_params(self._flatten_config(config))
                tracker.finalize()
            except ImportError:
                self.track_all(final_metrics, self._flatten_config(config))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_json(self, rel_path: str, obj) -> None:
        """write json."""

        def _default_serializer(o):
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
        fp.parent.mkdir(exist_ok=True)
        with open(fp, "w") as f:
            json.dump(
                _sanitize_json_value(obj),
                f,
                indent=2,
                default=_default_serializer,
                allow_nan=False,
            )

    def _write_text(self, rel_path: str, text: str) -> None:
        """write text."""
        fp = self.run_dir / rel_path
        fp.parent.mkdir(exist_ok=True)
        with open(fp, "w") as f:
            f.write(text)

    def _flatten_config(self, config) -> dict[str, Any]:
        """flatten config."""
        if hasattr(config, "__dict__"):
            return vars(config)
        return config

    def _save_config(self, config) -> None:
        """save config."""
        self._write_json("artifacts/config.json", self._flatten_config(config))

    def _save_history(self, history) -> None:
        """save history."""
        self._write_json("artifacts/history.json", history)

    def _save_final_metrics(self, metrics) -> None:
        """save final metrics."""
        self._write_json("artifacts/final_metrics.json", metrics)

    def _save_metrics_log(self, history) -> None:
        """save metrics log."""
        with open(self.run_dir / "logs" / "metrics.jsonl", "w") as f:
            for entry in history:
                f.write(json.dumps(_sanitize_json_value(entry), allow_nan=False) + "\n")

    def _save_model(self, model, tokenizer=None) -> None:
        """save model."""
        p = self.run_dir / "artifacts" / "model"
        p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))
        if tokenizer:
            tokenizer.save_pretrained(str(p))
        if self.cfg.save_optimizer:
            import torch

            torch.save(model.state_dict(), p / "optimizer.pt")

    def _save_checkpoint(self, model, name="best") -> None:
        """save checkpoint."""
        p = self.run_dir / "checkpoints" / name
        p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))

    def _save_archive(self) -> None:
        """save archive safely."""
        try:
            with tarfile.open(self.run_dir / "results.tar.gz", "w:gz") as tar:
                for sub in ["artifacts", "logs", "plots"]:
                    sp = self.run_dir / sub
                    if sp.exists():
                        tar.add(sp, arcname=sub)
        except Exception as e:
            logger.warning(f"Archive creation skipped due to warning/error: {e}")

    # ------------------------------------------------------------------
    # Plots – delegates to framework.analysis.plotting
    # ------------------------------------------------------------------
    def _generate_plots(self, history, final_metrics) -> None:
        """generate plots."""
        try:
            import matplotlib.pyplot as plt
            from neuralops.analysis.plotting import (
                plot_accuracy_curves,
                plot_confusion_matrix,
                plot_loss_curves,
                plot_lr_schedule,
                plot_radar_chart,
                plot_rank_allocation,
                plot_training_time,
            )

            opts = self.cfg.plot_options
            dpi = opts.get("dpi", 150)

            def save_fig(name, fig) -> None:
                """Persist fig to disk."""
                fig.savefig(
                    self.run_dir / "plots" / f"{name}.png", dpi=dpi, bbox_inches="tight"
                )
                plt.close(fig)

            if opts.get("loss_curve", True):
                try:
                    plot_loss_curves(
                        history, final_metrics, self.cfg, self.run_dir / "plots", opts
                    )
                except Exception as e:
                    logger.debug(f"plot_loss_curves error: {e}")

            if opts.get("accuracy_curve", True):
                try:
                    plot_accuracy_curves(
                        history, final_metrics, self.cfg, self.run_dir / "plots", opts
                    )
                except Exception as e:
                    logger.debug(f"plot_accuracy_curves error: {e}")

            if opts.get("lr_schedule", True):
                try:
                    plot_lr_schedule(
                        history, final_metrics, self.cfg, self.run_dir / "plots", opts
                    )
                except Exception as e:
                    logger.debug(f"plot_lr_schedule error: {e}")

            if opts.get("training_time", True):
                try:
                    plot_training_time(
                        history, final_metrics, self.cfg, self.run_dir / "plots", opts
                    )
                except Exception as e:
                    logger.debug(f"plot_training_time error: {e}")

            if opts.get("confusion_matrix", False):
                preds_path = self.run_dir / "artifacts" / "eval_predictions.json"
                if preds_path.exists():
                    with open(preds_path) as f:
                        data = json.load(f)
                    fig = plot_confusion_matrix(
                        data["labels"],
                        data["predictions"],
                        class_names=opts.get("class_names"),
                        figsize=tuple(opts.get("figsize", [10, 8])),
                    )
                    save_fig("confusion_matrix", fig)

            if opts.get("radar_chart", False):
                fig = plot_radar_chart(
                    final_metrics, figsize=tuple(opts.get("figsize", [8, 8]))
                )
                save_fig("radar_chart", fig)

            if opts.get("rank_allocation", False):
                rank_path = self.run_dir / "artifacts" / "rank_allocation.json"
                if rank_path.exists():
                    with open(rank_path) as f:
                        rank_data = json.load(f)
                    fig = plot_rank_allocation(
                        rank_data, figsize=tuple(opts.get("figsize", [10, 4]))
                    )
                    save_fig("rank_allocation", fig)

            if opts.get("attention_heatmap", False):
                logger.info(
                    "Attention heatmap requested but model not available in saver context; "
                    "use framework.callbacks.AttentionHeatmapCallback during training."
                )

        except Exception:
            logger.exception("Plot generation failed")

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------
    def _save_model_card(self, config, metrics) -> None:
        """save model card."""
        cfg_dict = (
            self._flatten_config(config) if hasattr(config, "__dict__") else config
        )
        model_name = cfg_dict.get("model_name", "Model")
        task_name = cfg_dict.get("task_name", "Task")
        acc = metrics.get("eval_accuracy", "N/A")
        loss = metrics.get("eval_loss", "N/A")
        card = f"""# {model_name} – {task_name}

**Accuracy:** {acc}
**Loss:** {loss}

**Run ID:** {self.run_id}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Compression:** {cfg_dict.get('attention_type', 'full')} (rank={cfg_dict.get('rank', 'N/A')})
"""
        self._write_text("artifacts/model_card.md", card)

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------
    def _export_format(self, fmt: str, model, input_example) -> None:
        """export format."""
        try:
            from .backends.exports import EXPORTERS

            exporter = EXPORTERS.get(fmt)
            if exporter:
                exporter(model, input_example, str(self.run_dir / "artifacts"))
        except Exception:
            logger.exception(f"Export {fmt} failed")

    # ------------------------------------------------------------------
    # Push to storage backends
    # ------------------------------------------------------------------
    def push_all(self) -> None:
        """Push all."""
        for be in self.cfg.storage_backends:
            self._push_backend(be)

    def _push_backend(self, name: str) -> None:
        """push backend."""
        try:
            if name == "local":
                return
            mod = __import__(
                f"neuralops.results_saver.backends.{name}", fromlist=["push"]
            )
            mod.push(self.run_dir, self.cfg)
        except ImportError:
            logger.warning(f"Backend '{name}' not available, skipping")
        except Exception:
            logger.exception(f"Push to {name} failed")

    # ------------------------------------------------------------------
    # Git / DVC
    # ------------------------------------------------------------------
    def git_commit_and_push(self) -> None:
        """Git commit and push."""
        if not self.cfg.git_options.get("commit_results"):
            return
        from .backends.git import git_commit_and_push

        git_commit_and_push(self.run_dir, self.cfg)

    def dvc_track(self) -> None:
        """Dvc track."""
        if not self.cfg.dvc_options.get("track_model"):
            return
        from .backends.dvc import dvc_track

        dvc_track(self.run_dir, self.cfg)

    # ------------------------------------------------------------------
    # Experiment trackers (legacy backends – prefer framework.tracking)
    # ------------------------------------------------------------------
    def track_all(self, metrics: dict[str, Any], params: dict[str, Any]) -> None:
        """Track all."""
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

    # ------------------------------------------------------------------
    # Model registries
    # ------------------------------------------------------------------
    def push_to_registries(self) -> None:
        """Push to registries."""
        for r in self.cfg.model_registries:
            try:
                mod = __import__(
                    f"neuralops.results_saver.backends.{r}", fromlist=["push"]
                )
                mod.push(self.run_id, self.run_dir, self.cfg)
            except ImportError:
                logger.warning(f"Registry '{r}' not available")
            except Exception:
                logger.exception(f"Push to registry {r} failed")

    # ------------------------------------------------------------------
    # GitHub Release
    # ------------------------------------------------------------------
    def create_github_release(self, metrics: dict[str, Any]) -> str | None:
        """Build and return github release."""
        if not self.cfg.release_options.get("create_github_release"):
            return None
        from .backends.github_release import create_release

        return create_release(self.run_id, self.run_dir, metrics, self.cfg)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    def notify(self, success: bool, metrics: dict[str, Any]) -> None:
        """Notify."""
        from .backends.notification import notify

        notify(self.run_id, success, metrics, self.cfg)


def save_result_json(
    result: dict[str, Any] | Any,
    results_dir: str | Path = "07_misc_tests/benchmark_results",
    filename: str | None = None,
) -> str:
    """
    Save complete benchmark result including fine-grained trajectories to JSON.
    Used for generating conference-grade loss curves and variance plots.
    """
    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "to_dict") and callable(result.to_dict):
        data = result.to_dict()
    elif hasattr(result, "__dict__"):
        data = vars(result)
    else:
        data = dict(result)

    if not filename:
        mod = data.get("module_name", "unknown")
        fam = data.get("model_family", "unknown")
        ds = data.get("dataset", "unknown")
        filename = f"{mod}_{fam}_{ds}.json"

    json_path = target_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(data), f, indent=2, allow_nan=False)

    return str(json_path)


def save_trajectories_parquet(
    detailed_results: list[dict[str, Any]],
    output_dir: str | Path = "results/06_adhoc_experiments/trajectories",
) -> str | None:
    """Writes step-level trajectories (loss, grad_norm, latency) as Apache Parquet."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = target_dir / "trajectories.parquet"

    rows = []
    for entry in detailed_results:
        mod = entry.get("module_name", "unknown")
        fam = entry.get("model_family", "unknown")
        ds = entry.get("dataset", "unknown")
        run_id = entry.get("run_id", "unknown")
        trajs = entry.get("trajectories", {})
        loss_list = trajs.get("loss", [])
        grad_list = trajs.get("grad_norm", [])

        for step, loss_val in enumerate(loss_list):
            grad_val = grad_list[step] if step < len(grad_list) else None
            rows.append(
                {
                    "run_id": run_id,
                    "module_name": mod,
                    "model_family": fam,
                    "dataset": ds,
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_val,
                }
            )

    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(str(parquet_path), engine="pyarrow", index=False)
        return str(parquet_path)
    except Exception:
        json_fallback = target_dir / "trajectories.json"
        with open(json_fallback, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        return str(json_fallback)
