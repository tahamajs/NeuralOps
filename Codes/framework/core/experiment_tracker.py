# framework/core/experiment_tracker.py
import os
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

class ExperimentTracker:
    def __init__(self, output_dir: str, run_name: Optional[str] = None):
        self.output_dir = output_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.run_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics: List[Dict[str, Any]] = []
        self._step_counter: Dict[str, int] = {}

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if step is None:
            step = self._step_counter.get(name, 0)
            self._step_counter[name] = step + 1
        entry = {"name": name, "value": float(value), "step": step, "timestamp": datetime.now().isoformat()}
        self.metrics.append(entry)
        with open(os.path.join(self.run_dir, "metrics_log.json"), "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def save_config(self, config: Dict[str, Any]):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary: Dict[str, Any]):
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

    def get_best_value(self, name: str, higher_is_better: bool = True) -> Optional[float]:
        values = [m["value"] for m in self.metrics if m["name"] == name]
        if not values:
            return None
        return max(values) if higher_is_better else min(values)

    def get_best_step(self, name: str, higher_is_better: bool = True) -> Optional[int]:
        values = [m for m in self.metrics if m["name"] == name]
        if not values:
            return None
        best = max(values, key=lambda x: x["value"]) if higher_is_better else min(values, key=lambda x: x["value"])
        return best["step"]

    def save_checkpoint(self, model, optimizer, epoch: int, additional: Optional[Dict] = None):
        checkpoint_dir = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        if additional:
            checkpoint.update(additional)
        torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pt"))

    def load_checkpoint(self, model, optimizer, epoch: int):
        path = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}", "checkpoint.pt")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def generate_report(self) -> str:
        report_path = os.path.join(self.run_dir, "report.html")
        html = f"""<!DOCTYPE html>
<html><head><title>Experiment Report: {self.run_name}</title>
<style>body{{font-family:Arial;margin:20px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{background:#f2f2f2}}</style>
</head><body><h1>Experiment Report: {self.run_name}</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
<h2>Summary</h2><table><thead><tr><th>Metric</th><th>Best Value</th><th>Step</th></tr></thead><tbody>"""
        for name in set(m["name"] for m in self.metrics):
            best = self.get_best_value(name)
            step = self.get_best_step(name)
            html += f"<tr><td>{name}</td><td>{best:.4f}</td><td>{step}</td></tr>"
        html += "</tbody></table></body></html>"
        with open(report_path, "w") as f:
            f.write(html)
        return report_path

def create_tracker(output_dir: str = "results", run_name: Optional[str] = None) -> ExperimentTracker:
    return ExperimentTracker(output_dir, run_name)