import os, sys, json, logging, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from django.conf import settings as django_settings
from .models import Experiment
from .jobs import push_update
import psutil
from django.utils import timezone

from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY
from framework.models.model_loader import load_model
from framework.data.dataset_loader import ExperimentConfig, load_dataset
from framework.data.preprocessing import preprocess_function
from framework.data.glue_loader import get_num_labels_and_is_regression, GLUE_TASK_KEYS
from framework.trainer.trainer import CompressionTrainer, get_compute_metrics
from framework.utils.device import get_device
from framework.utils.helpers import set_seed

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, TrainerCallback
)
import torch

logger = logging.getLogger(__name__)

def _log(exp, msg):
    exp.logs = (exp.logs or "") + msg + "\n"
    exp.save()

def _update_status(exp, status):
    exp.status = status
    exp.save()

class _MetricsCallback(TrainerCallback):
    """
    Calls push_update on every log and periodically sends progress.
    Also checks for cancellation via the stop_event.
    """
    def __init__(self, experiment: Experiment, total_steps: int, stop_event=None):
        self.experiment = experiment
        self.total_steps = total_steps
        self.stop_event = stop_event
        self.start_time = time.time()
        self.last_progress_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        # Check for cancellation signal
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
            _log(self.experiment, "Training cancelled by user.")
            _update_status(self.experiment, "cancelled")
            return

        # Only update progress every 10 steps to avoid flooding
        if state.global_step - self.last_progress_step < 10:
            return
        self.last_progress_step = state.global_step

        elapsed = time.time() - self.start_time
        self.experiment.current_step = state.global_step
        self.experiment.total_steps = self.total_steps

        if state.global_step > 0 and self.total_steps > 0:
            steps_per_sec = state.global_step / elapsed
            remaining_steps = self.total_steps - state.global_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            self.experiment.estimated_remaining_seconds = eta
        else:
            eta = 0

        # System metrics
        self.experiment.cpu_usage = psutil.cpu_percent()
        self.experiment.ram_usage_mb = psutil.virtual_memory().used / 1024**2
        self.experiment.save(update_fields=[
            'current_step', 'total_steps', 'estimated_remaining_seconds',
            'cpu_usage', 'ram_usage_mb'
        ])

        push_update(self.experiment, {
            "type": "progress",
            "current_step": state.global_step,
            "total_steps": self.total_steps,
            "eta_seconds": eta,
            "cpu": self.experiment.cpu_usage,
            "ram_mb": self.experiment.ram_usage_mb,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            push_update(self.experiment, {"type": "metrics", "data": logs})


def run_training(experiment: Experiment, stop_event=None):
    config = experiment.config
    _update_status(experiment, "running")
    _log(experiment, "Starting training pipeline...")

    device = get_device()
    set_seed(config.get("seed", 42))

    # 1. Model
    model_name = config.get("model_name", "distilbert-base-uncased")
    _log(experiment, f"Loading model: {model_name}")
    model, tokenizer = load_model(
        model_name=model_name,
        num_labels=None,
        attention_type=config.get("attention_type", "full"),
        rank=config.get("compression_rank", 64),
        device=device,
        use_unsloth=config.get("use_unsloth", False),
        cache_dir=config.get("cache_dir"),
    )

    # 2. Dataset
    task = config.get("task_name", "mrpc")
    _log(experiment, f"Loading dataset: {task}")
    data_cfg = ExperimentConfig(
        dataset_name=task,
        max_seq_length=config.get("max_seq_length", 128),
        split_ratio=config.get("split_ratio", 0.1),
        glue_base_path=config.get("glue_base_path", "./datasets/glue"),
        num_labels=None,
    )
    dataset_dict = load_dataset(data_cfg)
    num_labels, is_regression = get_num_labels_and_is_regression(task)
    if hasattr(model, 'config') and model.config.num_labels != num_labels:
        model.config.num_labels = num_labels

    # 3. Tokenization
    sent1_key, sent2_key = GLUE_TASK_KEYS.get(task, (None, None))
    preprocess_fn = preprocess_function(
        tokenizer=tokenizer,
        max_seq_length=config.get("max_seq_length", 128),
        sent1_key=sent1_key,
        sent2_key=sent2_key,
        is_regression=(task == "stsb"),
    )
    encoded = dataset_dict.map(preprocess_fn, batched=True,
                               remove_columns=dataset_dict["train"].column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in encoded["train"].column_names:
        cols.append("token_type_ids")
    encoded.set_format(type="torch", columns=cols)

    # 4. Output directory
    output_dir = os.path.join(str(django_settings.MEDIA_ROOT), str(experiment.id), "model")
    os.makedirs(output_dir, exist_ok=True)

    # 5. Estimate total training steps
    train_batch_size = config.get("batch_size", 16)
    epochs = config.get("epochs", 3)
    total_steps = (len(encoded["train"]) // train_batch_size) * epochs
    experiment.total_steps = total_steps
    experiment.save()

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size * 2,
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config.get("fp16", torch.cuda.is_available()),
        remove_unused_columns=False,
        report_to="none",
    )

    # 7. Trainer with callback
    metrics_callback = _MetricsCallback(experiment, total_steps, stop_event)
    trainer = CompressionTrainer(
        ortho_weight=0.0,
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation") or encoded.get("test"),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(task),
        callbacks=[metrics_callback],   # <-- pass directly
    )

    _log(experiment, "Training started...")
    trainer.train()

    # Check if cancelled before saving
    if stop_event and stop_event.is_set():
        _log(experiment, "Training stopped before completion. Model not saved.")
        return

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    final_metrics = trainer.evaluate()
    _log(experiment, f"Final metrics: {json.dumps(final_metrics)}")
    experiment.config["final_metrics"] = final_metrics
    experiment.save()


def run_evaluation(experiment: Experiment, stop_event=None):
    config = experiment.config
    _update_status(experiment, "running")
    _log(experiment, "Loading model for evaluation...")

    model_path = config.get("model_name")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = get_device()
    model.to(device)

    task = config.get("task_name", "mrpc")
    data_cfg = ExperimentConfig(dataset_name=task, max_seq_length=config.get("max_seq_length", 128))
    dataset_dict = load_dataset(data_cfg)
    sent1_key, sent2_key = GLUE_TASK_KEYS.get(task, (None, None))
    preprocess_fn = preprocess_function(
        tokenizer=tokenizer, max_seq_length=config.get("max_seq_length", 128),
        sent1_key=sent1_key, sent2_key=sent2_key, is_regression=(task=="stsb")
    )
    encoded = dataset_dict.map(preprocess_fn, batched=True,
                               remove_columns=dataset_dict["train"].column_names)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    from transformers import Trainer, TrainingArguments
    eval_args = TrainingArguments(output_dir="/tmp/eval",
                                 per_device_eval_batch_size=config.get("batch_size", 32))
    trainer = Trainer(model=model, args=eval_args,
                      eval_dataset=encoded.get("validation") or encoded.get("test"))
    results = trainer.evaluate()
    _log(experiment, f"Evaluation results: {json.dumps(results)}")
    experiment.config["final_metrics"] = results
    experiment.save()


def run_compression(experiment: Experiment, stop_event=None):
    config = experiment.config
    _update_status(experiment, "running")
    _log(experiment, "Loading model for compression...")

    model_name = config.get("model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = get_device()
    model.to(device)

    method = config.get("compression_method", "tt")
    compressor_cls = COMPRESSOR_REGISTRY.get(method)
    if not compressor_cls:
        raise ValueError(f"Unknown compression method: {method}")

    compressor = compressor_cls(model, rank=config.get("target_rank", 64))
    compressed_model, stats = compressor.compress()
    _log(experiment, f"Compression stats: {json.dumps(stats)}")

    output_dir = os.path.join(str(django_settings.MEDIA_ROOT), str(experiment.id), "model")
    os.makedirs(output_dir, exist_ok=True)
    compressed_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    experiment.config["stats"] = stats
    experiment.save()