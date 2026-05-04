import json
import csv
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def save_experiment_results(run, trainer, tokenizer, history, final_metrics, output_dir="results"):
    """Save all experiment outputs to a timestamped folder: JSON, CSV, PNG, model, logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{run.config.get('task_name', 'task')}_{run.config.get('model_name', 'model')}_{str(run.id)[:8]}_{timestamp}"
    save_path = Path(output_dir) / exp_name
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Config (JSON)
    with open(save_path / "config.json", "w") as f:
        json.dump(run.config, f, indent=2)

    # 2. Training history (JSON)
    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 3. Final metrics (JSON)
    with open(save_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # 4. Training history as CSV
    if history.get("steps") and history.get("losses"):
        with open(save_path / "training_history.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss"])
            for step, loss in zip(history["steps"], history["losses"]):
                writer.writerow([step, loss])
        if history.get("eval_steps") and history.get("eval_losses"):
            with open(save_path / "eval_history.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "eval_loss", "eval_accuracy"])
                for i, epoch in enumerate(history["eval_steps"]):
                    eval_loss = history["eval_losses"][i]
                    acc = history["eval_metrics"][i].get("eval_accuracy", "") if i < len(history["eval_metrics"]) else ""
                    writer.writerow([epoch, eval_loss, acc])

    # 5. Raw logs (text)
    if run.logs:
        with open(save_path / "logs.txt", "w") as f:
            f.write(run.logs)

    # 6. Loss plot (PNG)
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

    # 7. Model
    model_save_path = save_path / "model"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    # 8. Copy checkpoints (if any)
    if trainer.args.output_dir and Path(trainer.args.output_dir).exists():
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

    print(f"✅ Results saved to: {save_path}")
    return save_path