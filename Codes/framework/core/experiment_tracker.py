# framework/core/experiment_tracker.py
"""
Experiment tracking and logging utilities.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd


class ExperimentTracker:
    """
    Track experiment metrics, configurations, and results.
    
    This class handles logging of metrics, saving configurations,
    and generating summary reports for experiments.
    
    Example:
        >>> tracker = ExperimentTracker(output_dir="results", run_name="my_experiment")
        >>> tracker.save_config({"model": "bert", "rank": 64})
        >>> tracker.log_metric("accuracy", 0.92, step=100)
        >>> tracker.log_metric("loss", 0.23, step=100)
        >>> tracker.save_summary({"final_accuracy": 0.93})
    """
    
    def __init__(self, output_dir: str, run_name: Optional[str] = None):
        """
        Args:
            output_dir: Root directory for experiment results
            run_name: Name for this run (auto-generated if None)
        """
        self.output_dir = output_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.run_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.metrics: List[Dict[str, Any]] = []
        self._step_counter: Dict[str, int] = {}
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (auto-incremented if not provided)
        """
        if step is None:
            step = self._step_counter.get(name, 0)
            self._step_counter[name] = step + 1
        
        entry = {
            "name": name,
            "value": float(value),
            "step": step,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics.append(entry)
        
        # Append to JSON log file
        log_file = os.path.join(self.run_dir, "metrics_log.json")
        with open(log_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save final experiment summary.
        
        Args:
            summary: Summary dictionary
        """
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    def save_metrics_csv(self):
        """Save all logged metrics to a CSV file."""
        df = self.get_dataframe()
        if not df.empty:
            csv_path = os.path.join(self.run_dir, "metrics.csv")
            df.to_csv(csv_path, index=False)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get logged metrics as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: name, value, step, timestamp
        """
        return pd.DataFrame(self.metrics)
    
    def get_metrics_for(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all logged values for a specific metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of metric entries
        """
        return [m for m in self.metrics if m["name"] == name]
    
    def get_best_value(self, name: str, higher_is_better: bool = True) -> float:
        """
        Get the best value for a metric.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
            
        Returns:
            Best metric value
        """
        values = [m["value"] for m in self.metrics if m["name"] == name]
        if not values:
            return None
        
        if higher_is_better:
            return max(values)
        return min(values)
    
    def get_best_step(self, name: str, higher_is_better: bool = True) -> int:
        """
        Get the step at which the best value occurred.
        
        Args:
            name: Metric name
            higher_is_better: Whether higher values are better
            
        Returns:
            Step number
        """
        values = [m for m in self.metrics if m["name"] == name]
        if not values:
            return None
        
        if higher_is_better:
            best = max(values, key=lambda x: x["value"])
        else:
            best = min(values, key=lambda x: x["value"])
        
        return best["step"]
    
    def save_checkpoint(self, model, optimizer, epoch: int, additional: Optional[Dict] = None):
        """
        Save a model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            additional: Additional data to save
        """
        checkpoint_dir = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if additional:
            checkpoint.update(additional)
        
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, model, optimizer, epoch: int):
        """
        Load a model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Epoch to load
            
        Returns:
            Checkpoint data
        """
        checkpoint_path = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}", "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint
    
    def generate_report(self) -> str:
        """
        Generate a basic HTML report.
        
        Returns:
            Path to the generated HTML file
        """
        report_path = os.path.join(self.run_dir, "report.html")
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {self.run_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ margin: 10px 0; }}
        .good {{ color: green; font-weight: bold; }}
        .bad {{ color: red; }}
    </style>
</head>
<body>
    <h1>Experiment Report: {self.run_name}</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Best Value</th><th>Step</th></tr>
"""
        
        # Add best values for each metric
        metric_names = set(m["name"] for m in self.metrics)
        for name in metric_names:
            best = self.get_best_value(name)
            step = self.get_best_step(name)
            html += f"""        <tr><td>{name}</td><td>{best:.4f}</td><td>{step}</td></tr>\n"""
        
        html += """    </table>
</body>
</html>"""
        
        with open(report_path, "w") as f:
            f.write(html)
        
        return report_path
    
    def __repr__(self) -> str:
        return f"ExperimentTracker(run_name='{self.run_name}', dir='{self.run_dir}')"


# Convenience function
def create_tracker(output_dir: str = "results", run_name: Optional[str] = None) -> ExperimentTracker:
    """
    Create an experiment tracker with default settings.
    
    Args:
        output_dir: Output directory
        run_name: Run name (auto-generated if None)
        
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(output_dir, run_name)