import os
import sys
import json
import time
import traceback
import asyncio
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Allow importing framework modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from django.conf import settings as django_settings
from .models import Experiment
import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    Trainer,
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# ---------- WebSocket helper (works from any thread) ----------
from channels.layers import get_channel_layer

def send_ws_update(experiment_id, data):
    channel_layer = get_channel_layer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            channel_layer.group_send(
                f'job_{experiment_id}',
                {'type': 'job_update', 'data': data}
            )
        )
    except Exception as e:
        print(f"WebSocket send error: {e}")
    finally:
        loop.close()

# ---------- Logging helpers ----------
def _log(exp, msg):
    print(f"[LOG] {msg}")
    exp.logs = (exp.logs or "") + msg + "\n"
    exp.save()
    send_ws_update(str(exp.id), {"type": "log", "data": msg})

def _update_status(exp, status):
    print(f"[STATUS] {status}")
    exp.status = status
    exp.save()
    send_ws_update(str(exp.id), {"type": "status", "status": status})

# ---------- Metrics function ----------
def compute_metrics(task, pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    if task == "cola":
        return {"matthews_correlation": matthews_corrcoef(labels, preds)}
    elif task in ["mrpc", "qqp"]:
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}
    else:
        return {"accuracy": accuracy_score(labels, preds)}

# ---------- Live Callback (sends metrics with step) ----------
class LiveCallback(TrainerCallback):
    def __init__(self, exp, total_steps, stop_event):
        self.exp = exp
        self.total_steps = total_steps
        self.stop_event = stop_event
        self.start_time = time.time()
        self.last_progress = 0
        # History storage for disk saving
        self.history = {
            "steps": [],
            "losses": [],
            "eval_steps": [],
            "eval_losses": [],
            "eval_metrics": [],
        }
        self.eval_step_counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Record training loss per step
            if "loss" in logs:
                self.history["steps"].append(state.global_step)
                self.history["losses"].append(logs["loss"])
            # Record eval loss and metrics
            if "eval_loss" in logs:
                self.eval_step_counter += 1
                self.history["eval_steps"].append(self.eval_step_counter)  # epoch number
                self.history["eval_losses"].append(logs["eval_loss"])
                eval_metrics = {k: v for k, v in logs.items() if k.startswith("eval_")}
                self.history["eval_metrics"].append(eval_metrics)

            # Build metrics dictionary for frontend
            metrics = {}
            if "loss" in logs:
                metrics["loss"] = logs["loss"]
                metrics["step"] = state.global_step          # ← CRITICAL for x‑axis
            if "eval_loss" in logs:
                metrics["eval_loss"] = logs["eval_loss"]
            if "eval_accuracy" in logs:
                metrics["eval_accuracy"] = logs["eval_accuracy"]
            if "learning_rate" in logs:
                metrics["learning_rate"] = logs["learning_rate"]

            if metrics:
                send_ws_update(str(self.exp.id), {"type": "metrics", "data": metrics})

            # Also send raw log for terminal display
            send_ws_update(str(self.exp.id), {"type": "log", "data": f"Step {state.global_step}: {logs}"})

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
            return
        # Update progress every 5 steps to avoid flooding
        if state.global_step - self.last_progress >= 5:
            self.last_progress = state.global_step
            elapsed = time.time() - self.start_time
            pct = (state.global_step / self.total_steps) * 100 if self.total_steps else 0
            eta = (self.total_steps - state.global_step) * (elapsed / max(1, state.global_step))
            self.exp.current_step = state.global_step
            self.exp.estimated_remaining_seconds = eta
            self.exp.cpu_usage = psutil.cpu_percent()
            self.exp.ram_usage_mb = psutil.virtual_memory().used / 1024**2
            self.exp.save(update_fields=[
                'current_step', 'estimated_remaining_seconds', 'cpu_usage', 'ram_usage_mb'
            ])
            send_ws_update(str(self.exp.id), {
                "type": "progress",
                "progress": pct,
                "current_step": state.global_step,
                "total_steps": self.total_steps,
                "eta_seconds": eta,
            })

# ---------- Results saving function ----------
def save_experiment_results(exp, trainer, tokenizer, history, final_metrics, output_dir="results"):
    """Save all experiment outputs to a timestamped folder."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{exp.config.get('task_name', 'task')}_{exp.config.get('model_name', 'model')}_{str(exp.id)[:8]}_{timestamp}"
    save_path = Path(output_dir) / exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Save configuration
    with open(save_path / "config.json", "w") as f:
        json.dump(exp.config, f, indent=2)

    # 2. Save training history
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 3. Save final metrics
    with open(save_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # 4. Save raw logs
    if exp.logs:
        with open(save_path / "logs.txt", "w") as f:
            f.write(exp.logs)

    # 5. Plot loss curves
    if history.get("steps") and history.get("losses"):
        plt.figure(figsize=(10, 6))
        plt.plot(history["steps"], history["losses"], label="Train Loss")
        if history.get("eval_steps") and history.get("eval_losses"):
            plt.plot(history["eval_steps"], history["eval_losses"], 'ro-', label="Eval Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Save model
    model_save_path = save_path / "model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    # 7. Copy checkpoints if any
    if trainer.args.output_dir and os.path.exists(trainer.args.output_dir):
        checkpoint_dir = save_path / "checkpoints"
        shutil.copytree(trainer.args.output_dir, checkpoint_dir, dirs_exist_ok=True)

    # 8. README
    with open(save_path / "README.txt", "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Task: {exp.config.get('task_name')}\n")
        f.write(f"Model: {exp.config.get('model_name')}\n")
        first_metric = list(final_metrics.keys())[0] if final_metrics else "eval_loss"
        first_value = list(final_metrics.values())[0] if final_metrics else "N/A"
        f.write(f"Final {first_metric}: {first_value}\n")
        f.write(f"Saved at: {datetime.now()}\n")

    print(f"✅ Results saved to: {save_path}")
    return save_path

# ---------- Local GLUE loader ----------
def load_glue_local(task_name, base_path):
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local GLUE path not found: {task_path}")
    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        lower = fname.lower()
        if lower.startswith("train"):
            split = "train"
        elif lower.startswith("validation"):
            split = "validation"
        elif lower.startswith("test"):
            split = "test"
        else:
            split = fname.replace(".parquet", "")
        splits[split] = Dataset.from_parquet(os.path.join(task_path, fname))
    return DatasetDict(splits)

# ---------- MAIN TRAINING ----------
def run_training(experiment: Experiment, stop_event=None):
    print("🔥 TRAINING THREAD STARTED 🔥")
    try:
        config = experiment.config
        _update_status(experiment, "running")
        _log(experiment, "Starting training pipeline...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(experiment, f"Device: {device}")

        model_name = config.get("model_name", "distilbert-base-uncased")
        task = config.get("task_name", "mrpc")
        _log(experiment, f"Model: {model_name}, Task: {task}")

        # Load tokenizer and model
        _log(experiment, "Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if task == "stsb":
            num_labels = 1
        elif task == "mnli":
            num_labels = 3
        else:
            num_labels = 2
        # IMPORTANT: output_attentions=True is not needed for training; keep for possible attention extraction
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)
        _log(experiment, "Model loaded")

        # Load local GLUE dataset
        glue_path = config.get("glue_base_path", "./datasets/glue")
        if not os.path.exists(glue_path):
            glue_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
        _log(experiment, f"Loading dataset from {glue_path}/{task}")
        dataset = load_glue_local(task, glue_path)

        available_splits = list(dataset.keys())
        _log(experiment, f"Available dataset splits: {available_splits}")

        if 'train' not in dataset:
            raise KeyError(f"No 'train' split found. Available: {available_splits}")
        train_dataset = dataset['train']
        val_dataset = dataset.get('validation', None)
        _log(experiment, f"Using 'train' split. Size: {len(train_dataset)}")
        if val_dataset:
            _log(experiment, f"Using 'validation' split. Size: {len(val_dataset)}")
        else:
            _log(experiment, "No validation split found. Evaluation will be skipped.")

        # Preprocessing
        def preprocess(examples):
            if task in ["cola", "sst2"]:
                return tokenizer(
                    examples['sentence'],
                    truncation=True,
                    padding='max_length',
                    max_length=config.get("max_seq_length", 128)
                )
            else:
                return tokenizer(
                    examples['sentence1'],
                    examples.get('sentence2', ''),
                    truncation=True,
                    padding='max_length',
                    max_length=config.get("max_seq_length", 128)
                )
        _log(experiment, "Tokenizing dataset...")
        encoded = dataset.map(preprocess, batched=True)
        encoded = encoded.rename_column("label", "labels")
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        train_encoded = encoded['train']
        val_encoded = encoded['validation'] if val_dataset else None
        _log(experiment, f"Tokenization done. Train size: {len(train_encoded)}")

        output_dir = os.path.join(str(django_settings.MEDIA_ROOT), str(experiment.id), "model")
        os.makedirs(output_dir, exist_ok=True)

        train_batch_size = config.get("batch_size", 16)
        epochs = config.get("epochs", 3)
        total_steps = (len(train_encoded) // train_batch_size) * epochs
        experiment.total_steps = total_steps
        experiment.save()
        _log(experiment, f"Total training steps: {total_steps}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size * 2,
            learning_rate=config.get("learning_rate", 2e-5),
            warmup_steps=500,
            logging_steps=1,
            eval_strategy="epoch" if val_encoded else "no",
            save_strategy="epoch" if val_encoded else "no",
            load_best_model_at_end=bool(val_encoded),
            metric_for_best_model="eval_loss" if val_encoded else None,
            fp16=False,
            report_to="none",
        )

        live_callback = LiveCallback(experiment, total_steps, stop_event)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=val_encoded,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=lambda pred: compute_metrics(task, pred) if val_encoded else None,
            callbacks=[live_callback],
        )

        _log(experiment, "Training started...")
        trainer.train()

        if stop_event and stop_event.is_set():
            _log(experiment, "Training cancelled by user.")
            _update_status(experiment, "stopped")
            return

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        final = trainer.evaluate() if val_encoded else {"message": "No evaluation dataset"}
        _log(experiment, f"Final metrics: {json.dumps(final)}")
        experiment.config["final_metrics"] = final
        experiment.save()
        _update_status(experiment, "completed")
        _log(experiment, "Training completed successfully!")

        # ✅ Update status to "completed" BEFORE heavy evaluation
        _update_status(experiment, "completed")
        _log(experiment, "Training completed, running final evaluation...")

        # Now run evaluation (can be long)
        final = trainer.evaluate() if val_encoded else {"message": "No evaluation dataset"}
        _log(experiment, f"Final metrics: {json.dumps(final)}")
        experiment.config["final_metrics"] = final
        experiment.save()

        # Save results to disk
        try:
            save_experiment_results(
                exp=experiment,
                trainer=trainer,
                tokenizer=tokenizer,
                history=live_callback.history,
                final_metrics=final,
                output_dir="results"
            )
        except Exception as e:
            _log(experiment, f"Warning: Could not save results to disk: {e}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"❌ TRAINING CRASHED:\n{error_msg}")
        _log(experiment, f"ERROR: {str(e)}")
        _log(experiment, f"TRACEBACK: {error_msg}")
        experiment.status = "failed"
        experiment.save()
        send_ws_update(str(experiment.id), {"type": "log", "data": f"FATAL ERROR: {str(e)}"})
        _update_status(experiment, "failed")
        if stop_event:
            stop_event.set()

# ---------- EVALUATION (unchanged) ----------
def run_evaluation(experiment: Experiment, stop_event=None):
    try:
        config = experiment.config
        _update_status(experiment, "running")
        _log(experiment, "Starting evaluation...")
        task = config.get("task_name", "mrpc")

        # Try to load fine-tuned model from experiment's saved directory
        model_dir = os.path.join(str(django_settings.MEDIA_ROOT), str(experiment.id), "model")
        if os.path.exists(model_dir):
            _log(experiment, f"Loading fine-tuned model from {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            _log(experiment, "No fine-tuned model found, using base model")
            model_name = config.get("model_name", "distilbert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load dataset (same as before)
        glue_path = config.get("glue_base_path", "./datasets/glue")
        if not os.path.exists(glue_path):
            glue_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
        dataset = load_glue_local(task, glue_path)

        if 'validation' not in dataset and 'test' not in dataset:
            raise KeyError(f"No evaluation split found. Available: {list(dataset.keys())}")
        val_key = 'validation' if 'validation' in dataset else 'test'

        def preprocess(examples):
            if task in ["cola", "sst2"]:
                return tokenizer(examples['sentence'], truncation=True, padding='max_length',
                                 max_length=config.get("max_seq_length", 128))
            else:
                return tokenizer(examples['sentence1'], examples.get('sentence2', ''),
                                 truncation=True, padding='max_length',
                                 max_length=config.get("max_seq_length", 128))

        encoded = dataset.map(preprocess, batched=True)
        encoded = encoded.rename_column("label", "labels")
        encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset = encoded[val_key]

        eval_args = TrainingArguments(
            output_dir="/tmp/eval",
            per_device_eval_batch_size=config.get("batch_size", 32),
            report_to="none",
        )
        trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset)
        results = trainer.evaluate()
        _log(experiment, f"Evaluation results: {json.dumps(results)}")
        experiment.config["final_metrics"] = results
        experiment.save()
        _update_status(experiment, "completed")
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Evaluation failed: {error_msg}")
        _log(experiment, f"Evaluation error: {str(e)}")
        experiment.status = "failed"
        experiment.save()
        _update_status(experiment, "failed")
        if stop_event:
            stop_event.set()
            
            # ---------- COMPRESSION (unchanged) ----------
def run_compression(experiment: Experiment, stop_event=None):
    try:
        config = experiment.config
        _update_status(experiment, "running")
        _log(experiment, "Starting compression...")
        model_name = config.get("model_name", "distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        try:
            from framework.compression.core.registry import COMPRESSOR_REGISTRY
        except ImportError:
            COMPRESSOR_REGISTRY = {}
        method = config.get("compression_method", "tt")
        compressor_cls = COMPRESSOR_REGISTRY.get(method)
        if compressor_cls is None:
            _log(experiment, f"Compression method '{method}' not found – using dummy (no compression).")
            _update_status(experiment, "completed")
            return
        compressor = compressor_cls(model, rank=config.get("target_rank", 64))
        _log(experiment, f"Compressing using {method}...")
        compressed_model, stats = compressor.compress()
        _log(experiment, f"Compression stats: {json.dumps(stats)}")
        output_dir = os.path.join(str(django_settings.MEDIA_ROOT), str(experiment.id), "model")
        os.makedirs(output_dir, exist_ok=True)
        compressed_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        experiment.config["stats"] = stats
        experiment.save()
        _update_status(experiment, "completed")
        _log(experiment, "Compression finished.")
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"Compression failed: {error_msg}")
        _log(experiment, f"Compression error: {str(e)}")
        experiment.status = "failed"
        experiment.save()
        _update_status(experiment, "failed")
        if stop_event:
            stop_event.set()