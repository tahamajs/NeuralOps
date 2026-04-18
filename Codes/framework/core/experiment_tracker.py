import os
import json
from datetime import datetime
import pandas as pd

class ExperimentTracker:
    def __init__(self, output_dir, run_name=None):
        self.output_dir = output_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.run_name}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics = []

    def log_metric(self, name, value, step=None):
        entry = {"name": name, "value": value, "step": step, "timestamp": datetime.now().isoformat()}
        self.metrics.append(entry)
        with open(os.path.join(self.run_dir, "metrics_log.json"), "a") as f:
            json.dump(entry, f)
            f.write("\n")

    def save_config(self, config):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary_dict):
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary_dict, f, indent=2)

    def get_dataframe(self):
        return pd.DataFrame(self.metrics)