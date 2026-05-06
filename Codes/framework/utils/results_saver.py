import os, json, torch, shutil
from pathlib import Path
from datetime import datetime

class ResultsSaver:
    def __init__(self, run_id, base_dir):
        self.run_dir = Path(base_dir) / str(run_id)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.exports_dir = self.run_dir / "exports"

    def create_structure(self):
        for d in [self.checkpoints_dir, self.logs_dir, self.plots_dir,
                  self.artifacts_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_config(self, config):
        with open(self.artifacts_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def save_metrics(self, metrics, step=None):
        # Append to metrics.jsonl
        record = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
        with open(self.logs_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(record) + "\n")

    def save_history(self, history):
        with open(self.artifacts_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

    def save_final_metrics(self, metrics):
        with open(self.artifacts_dir / "final_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_model(self, model, tokenizer=None):
        model.save_pretrained(self.checkpoints_dir / "final")
        if tokenizer:
            tokenizer.save_pretrained(self.checkpoints_dir / "final")

    def export_onnx(self, model, input_example):
        torch.onnx.export(model, input_example, self.exports_dir / "model.onnx")

    def export_torchscript(self, model):
        traced = torch.jit.trace(model, torch.randn(1, 128).long())
        traced.save(self.exports_dir / "model_torchscript.pt")

    def generate_plots(self, history):
        from framework.analysis.plotting import create_comprehensive_plots
        create_comprehensive_plots(history, str(self.plots_dir))

    def generate_model_card(self, config, metrics):
        card = f"# {config.get('model_name')} – {config.get('task_name')}\n"
        card += f"Accuracy: {metrics.get('eval_accuracy')}\n"
        card += f"Loss: {metrics.get('eval_loss')}\n"
        with open(self.artifacts_dir / "model_card.md", 'w') as f:
            f.write(card)

    def register_version(self, run, metrics):
        from model_versioning.services import ModelVersionService
        ModelVersionService.create_version(
            run=run,
            model_state=torch.load(self.checkpoints_dir / "final" / "pytorch_model.bin"),
            metrics=metrics,
            artifact_base=str(self.run_dir),
        )