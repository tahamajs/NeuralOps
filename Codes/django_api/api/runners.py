import os
import sys
import json
import time
import traceback
import asyncio
import shutil
import matplotlib
matplotlib.use('Agg')   # must be before importing pyplot
import matplotlib.pyplot as plt

from pathlib import Path
import matplotlib.pyplot as plt
from asgiref.sync import async_to_sync

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
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
# ---------- Logging helpers ----------
def _log(run, msg):
    print(f"[LOG] {msg}")
    run.logs = (run.logs or "") + msg + "\n"
    run.save()
    # Send to job‑specific WebSocket
    send_ws_update(str(run.id), {"type": "log", "data": msg})
    # Also send to terminal group (for xterm.js terminal)
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "terminal",
            {"type": "terminal_message", "data": msg}
        )
    except Exception as e:
        print(f"Terminal broadcast error: {e}")
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
def save_experiment_results(run, trainer, tokenizer, history, final_metrics, output_dir="results"):
    from datetime import datetime
    import shutil
    import json
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from django.conf import settings as django_settings

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{run.config.get('task_name', 'task')}_{run.config.get('model_name', 'model')}_{str(run.id)[:8]}_{timestamp}"
    save_path = Path(output_dir) / exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Config
    with open(save_path / "config.json", "w") as f:
        json.dump(run.config, f, indent=2)

    # 2. Training history (JSON)
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 3. Final metrics
    with open(save_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # 4. Raw logs
    if run.logs:
        with open(save_path / "logs.txt", "w") as f:
            f.write(run.logs)

    # 5. CSV exports
    if history.get("steps") and history.get("losses"):
        df = pd.DataFrame({'step': history['steps'], 'train_loss': history['losses']})
        df.to_csv(save_path / "training_history.csv", index=False)
        if history.get('eval_steps') and history.get('eval_losses'):
            eval_df = pd.DataFrame({
                'epoch': history['eval_steps'],
                'eval_loss': history['eval_losses']
            })
            eval_df.to_csv(save_path / "eval_metrics.csv", index=False)

    # 6. Loss curve plot
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
        plot_path = save_path / "loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Model
    model_save_path = save_path / "model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    # 8. Checkpoints (if any)
    if trainer.args.output_dir and os.path.exists(trainer.args.output_dir):
        checkpoint_dir = save_path / "checkpoints"
        shutil.copytree(trainer.args.output_dir, checkpoint_dir, dirs_exist_ok=True)

    # 9. README
    with open(save_path / "README.txt", "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Task: {run.config.get('task_name')}\n")
        f.write(f"Model: {run.config.get('model_name')}\n")
        first_metric = list(final_metrics.keys())[0] if final_metrics else "eval_loss"
        first_value = list(final_metrics.values())[0] if final_metrics else "N/A"
        f.write(f"Final {first_metric}: {first_value}\n")
        f.write(f"Saved at: {datetime.now()}\n")

    # 10. Copy all artifacts and model to MEDIA_ROOT/run.id/ for frontend artifacts list
    media_dir = Path(django_settings.MEDIA_ROOT) / str(run.id)
    media_dir.mkdir(parents=True, exist_ok=True)
    # Copy loss plot (already created) – ensure it exists
    if plot_path.exists():
        shutil.copy(plot_path, media_dir / "loss_plot.png")
    # Copy model directory contents into media_dir/model/
    model_media_dir = media_dir / "model"
    if model_media_dir.exists():
        shutil.rmtree(model_media_dir)
    shutil.copytree(model_save_path, model_media_dir)
    # Copy checkpoints (if any) into media_dir/checkpoints/
    checkpoint_media_dir = media_dir / "checkpoints"
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        if checkpoint_media_dir.exists():
            shutil.rmtree(checkpoint_media_dir)
        shutil.copytree(checkpoint_dir, checkpoint_media_dir)
    # Copy CSV and JSON files for direct access
    for f in save_path.glob("*.csv"):
        shutil.copy(f, media_dir / f.name)
    for f in save_path.glob("*.json"):
        shutil.copy(f, media_dir / f.name)
    # Copy logs.txt
    logs_file = save_path / "logs.txt"
    if logs_file.exists():
        shutil.copy(logs_file, media_dir / "logs.txt")
    # Also copy the training history JSON (already copied by *.json)
    # Optionally, create a symlink to the timestamped folder (but copying is simpler)

    print(f"✅ Results saved to: {save_path}")
    print(f"📁 Artifacts also available in media folder: {media_dir}")
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

        device = get_device()
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
        # Check for existing checkpoints
        checkpoint_dir = None
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by step number (e.g., checkpoint-123)
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                checkpoint_dir = os.path.join(output_dir, checkpoints[-1])
                _log(run, f"Found checkpoint, will resume from {checkpoint_dir}")




        train_batch_size = config.get("batch_size", 16)
        epochs = config.get("epochs", 3)
        total_steps = (len(train_encoded) // train_batch_size) * epochs
        experiment.total_steps = total_steps
        experiment.save()
        _log(experiment, f"Total training steps: {total_steps}")
        # Limit training samples
        max_train = config.get("max_train_samples", -1)
        if max_train > 0 and len(train_encoded) > max_train:
            train_encoded = train_encoded.select(range(max_train))
            _log(experiment, f"Limited training set to {max_train} samples")

        # Limit validation samples
        max_eval = config.get("max_eval_samples", -1)
        if val_encoded is not None and max_eval > 0 and len(val_encoded) > max_eval:
            val_encoded = val_encoded.select(range(max_eval))
            _log(experiment, f"Limited validation set to {max_eval} samples")



        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size * 2,
            learning_rate=config.get("learning_rate", 2e-5),
            warmup_steps=500,
            logging_steps=1,
            save_total_limit=2,  # keep only last 2 checkpoints

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
        trainer.train(resume_from_checkpoint=checkpoint_dir)

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
                run=experiment,
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