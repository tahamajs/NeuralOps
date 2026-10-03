"""
ResultWriter — Organised result persistence for every pipeline run.

Directory layout created per run
---------------------------------
<output_dir>/
├── config.json             — full ExperimentConfig as dict
├── metrics/
│   ├── eval_final.json     — final eval metrics
│   ├── compression.json    — compression stats (if applicable)
│   └── standalone_eval.json
├── history/
│   └── log_history.json    — full HuggingFace trainer log history
├── plots/
│   ├── loss_curve.png
│   └── metric_curve.png
├── system_info.json        — device, elapsed time, param count, Python ver
└── SUMMARY.md              — human-readable run summary
"""

import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultWriter:
    """Centralised, organised result persistence for a single pipeline run."""

    def __init__(self, output_dir: Path | str) -> None:
        """Initialise and create the standard directory tree under output_dir."""
        self.root = Path(output_dir)
        self._dirs = {
            "metrics": self.root / "metrics",
            "history": self.root / "history",
            "plots": self.root / "plots",
        }
        for d in self._dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Config                                                               #
    # ------------------------------------------------------------------ #
    def save_config(self, cfg: Any) -> None:
        """Serialise the ExperimentConfig (dataclass or dict) to config.json."""
        if hasattr(cfg, "to_dict"):
            data = cfg.to_dict()
        elif hasattr(cfg, "__dict__"):
            data = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        elif isinstance(cfg, dict):
            data = cfg
        else:
            data = {"config": str(cfg)}
        self._write_json(self.root / "config.json", data)
        logger.debug("Config saved → %s", self.root / "config.json")

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    def save_metrics(self, metrics: dict[str, Any], tag: str = "metrics") -> None:
        """Save a metrics dict to metrics/<tag>.json with a timestamp."""
        payload = {"timestamp": datetime.now().isoformat(), **metrics}
        path = self._dirs["metrics"] / f"{tag}.json"
        self._write_json(path, payload)
        logger.debug("Metrics[%s] saved → %s", tag, path)

    # ------------------------------------------------------------------ #
    # Training history                                                     #
    # ------------------------------------------------------------------ #
    def save_history(self, log_history: list[dict[str, Any]]) -> None:
        """Save the full HuggingFace trainer log_history list."""
        path = self._dirs["history"] / "log_history.json"
        self._write_json(path, log_history)

        # Also export as CSV for easy analysis
        csv_path = self._dirs["history"] / "log_history.csv"
        if log_history:
            keys = sorted({k for entry in log_history for k in entry})
            try:
                import csv

                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(log_history)
            except Exception as e:
                logger.warning("Could not write CSV history: %s", e)

    # ------------------------------------------------------------------ #
    # System info                                                          #
    # ------------------------------------------------------------------ #
    def save_system_info(
        self,
        elapsed_s: float = 0.0,
        n_params: int = 0,
        device: str = "cpu",
    ) -> None:
        """Record hardware, runtime, and parameter count metadata."""
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
            mps_available = torch.backends.mps.is_available()
        except Exception:
            cuda_available = mps_available = False
            gpu_name = "N/A"

        info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "gpu_name": gpu_name,
            "trainable_params": n_params,
            "elapsed_seconds": round(elapsed_s, 2),
            "elapsed_human": _human_time(elapsed_s),
        }
        self._write_json(self.root / "system_info.json", info)

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    def save_plots(self, log_history: list[dict[str, Any]]) -> None:
        """Generate and save training/validation loss and metric curves."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available — skipping plot generation")
            return

        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []
        eval_accs, eval_acc_steps = [], []

        for entry in log_history:
            step = entry.get("step", 0)
            if "loss" in entry:
                train_steps.append(step)
                train_losses.append(entry["loss"])
            if "eval_loss" in entry:
                eval_steps.append(step)
                eval_losses.append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                eval_acc_steps.append(step)
                eval_accs.append(entry["eval_accuracy"])

        # Loss curve
        if train_losses:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train_steps, train_losses, label="Train Loss", linewidth=1.5)
            if eval_losses:
                ax.plot(eval_steps, eval_losses, "o-", label="Eval Loss", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training / Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self._dirs["plots"] / "loss_curve.png", dpi=150)
            plt.close(fig)

        # Accuracy curve
        if eval_accs:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(
                eval_acc_steps, eval_accs, "g^-", label="Eval Accuracy", linewidth=1.5
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy")
            ax.set_title("Validation Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self._dirs["plots"] / "metric_curve.png", dpi=150)
            plt.close(fig)

    # ------------------------------------------------------------------ #
    # Summary Markdown                                                     #
    # ------------------------------------------------------------------ #
    def write_summary_md(
        self,
        exp_name: str,
        cfg: Any,
        metrics: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        """Write a human-readable SUMMARY.md for quick inspection."""
        lines = [
            f"# Run Summary: `{exp_name}`",
            f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Configuration",
            "```",
        ]
        if hasattr(cfg, "to_dict"):
            cfg_dict = cfg.to_dict()
        elif hasattr(cfg, "__dict__"):
            cfg_dict = vars(cfg)
        else:
            cfg_dict = cfg if isinstance(cfg, dict) else {}
        for k, v in sorted(cfg_dict.items()):
            if not str(k).startswith("_"):
                lines.append(f"  {k}: {v}")
        lines += ["```\n", "## Results\n", "| Metric | Value |", "|:---|:---|"]
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.6f} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines += [
            f"\n**Total runtime:** {_human_time(elapsed_s)}",
            f"\n**Output dir:** `{self.root}`",
        ]
        (self.root / "SUMMARY.md").write_text("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _write_json(self, path: Path, data: Any) -> None:
        """Write data to a JSON file with indentation."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def _human_time(seconds: float) -> str:
    """Convert seconds to a human-readable duration string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
