import os
import json
import torch
import shutil
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    HUGGINGFACE = "huggingface"

class ResultsSaver:
    def __init__(self, run_id: str, base_dir: Path, save_config: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / str(run_id)
        self.cfg = save_config or {
            "save_model": True,
            "save_checkpoints": True,
            "save_optimizer_state": True,
            "save_logs": True,
            "save_plots": True,
            "save_as_version": True,
            "export_onnx": False,
            "export_torchscript": False,
            "storage_backend": "local",
            "auto_save_if_better_than": None,
        }
        self._init_dirs()

    def _init_dirs(self):
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        if self.cfg.get("save_plots"):
            (self.run_dir / "plots").mkdir(parents=True, exist_ok=True)
        if any([self.cfg.get("save_model"), self.cfg.get("export_onnx"),
                self.cfg.get("export_torchscript")]):
            (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    def should_save_model(self, final_metrics: Dict[str, Any]) -> bool:
        threshold = self.cfg.get("auto_save_if_better_than")
        if threshold is None:
            return self.cfg.get("save_model", True)
        metric = final_metrics.get("eval_accuracy", 0)
        if metric >= threshold:
            logger.info(f"Auto‑save triggered: {metric} >= {threshold}")
            return True
        logger.info(f"Auto‑save skipped: {metric} < {threshold}")
        return False

    def save_metrics(self, metrics: Dict[str, Any], step: int = None):
        if not self.cfg.get("save_logs", True):
            return
        record = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
        with open(self.run_dir / "logs" / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(record) + "\n")

    def save_history(self, history):
        if not self.cfg.get("save_logs", True):
            return
        with open(self.run_dir / "artifacts" / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

    def save_final_metrics(self, metrics):
        with open(self.run_dir / "artifacts" / "final_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_config(self, config):
        with open(self.run_dir / "artifacts" / "config.json", 'w') as f:
            json.dump(vars(config) if hasattr(config, '__dict__') else config, f, indent=2)

    def save_model(self, model, tokenizer=None):
        if not self.cfg.get("save_model", True):
            return
        output = self.run_dir / "artifacts" / "model"
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output))
        if tokenizer:
            tokenizer.save_pretrained(str(output))
        if self.cfg.get("save_optimizer_state"):
            torch.save(model.state_dict(), output / "optimizer.pt")

    def export_onnx(self, model, input_example):
        if not self.cfg.get("export_onnx", False):
            return
        torch.onnx.export(model, input_example, str(self.run_dir / "artifacts" / "model.onnx"))

    def export_torchscript(self, model, input_example):
        if not self.cfg.get("export_torchscript", False):
            return
        traced = torch.jit.trace(model, input_example)
        traced.save(str(self.run_dir / "artifacts" / "model.pt"))

    def generate_plots(self, history):
        if not self.cfg.get("save_plots", True):
            return
        from framework.analysis.plotting import create_comprehensive_plots
        create_comprehensive_plots(history, str(self.run_dir / "plots"))

    def register_version(self, run, metrics):
        if not self.cfg.get("save_as_version", True):
            return
        from model_versioning.services import ModelVersionService
        state_path = self.run_dir / "artifacts" / "model" / "pytorch_model.bin"
        if state_path.exists():
            state = torch.load(state_path, map_location='cpu')
            ModelVersionService.create_version(
                run=run,
                model_state=state,
                metrics=metrics,
                artifact_base=str(self.base_dir),
            )

    def handle_storage_backend(self):
        backend = self.cfg.get("storage_backend", "local")
        if backend == "s3":
            self._upload_to_s3()
        elif backend == "huggingface":
            self._push_to_hub()

    def _upload_to_s3(self):
        # use boto3 to sync the entire run_dir
        import boto3
        bucket = os.environ.get("MODEL_ARCHIVE_BUCKET")
        if not bucket:
            logger.warning("S3 bucket not configured")
            return
        s3 = boto3.client('s3')
        for root, _, files in os.walk(self.run_dir):
            for f in files:
                local = os.path.join(root, f)
                key = os.path.relpath(local, self.base_dir)
                s3.upload_file(local, bucket, key)

    def _push_to_hub(self):
        from huggingface_hub import HfApi
        api = HfApi()
        repo = f"{os.environ.get('HUGGINGFACE_USERNAME')}/model-{self.run_id}"
        api.create_repo(repo, exist_ok=True)
        api.upload_folder(folder_path=str(self.run_dir), repo_id=repo)