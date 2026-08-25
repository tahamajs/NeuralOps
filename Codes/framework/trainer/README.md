# Trainer Subsystem (`Codes/framework/trainer`)

The `trainer` subsystem delivers training loops, fine-tuning paradigms, distributed synchronization, reinforcement learning algorithms, and parameter-efficient fine-tuning (PEFT) methods.

---

## 🏗 Architecture Overview

- **HuggingFace Integration (`hf_trainer.py`, `trainer.py`)**: `SVDAwareTrainer` extends HuggingFace `Trainer` with pre/post-training SVD callbacks, live attention map extraction, and automatic metric resolution.
- **High-Throughput Custom Loop (`custom_loop_trainer.py`)**: Native PyTorch training loop featuring mixed precision (AMP FP16/BF16), gradient accumulation without redundant DDP synchronization, and factorized gradient clipping.
- **Reinforcement Learning (`rl_trainer.py`)**: `RLTrainer` supporting Direct Preference Optimization (DPO), Proximal Policy Optimization (PPO), and native policy-gradient optimization.
- **Fast-LoRA Kernel Acceleration (`unsloth_trainer.py`)**: Integration with Unsloth fast kernels with fallback to standard HuggingFace training.
- **PEFT Methods Library (`methods/`)**: Implementations of LoRA, GeLoRA, Layer-wise LR Decay (LLRD), Mixout, R-Drop / R3F regularization, and Fisher Information weighting.
- **Factory & Registry (`trainer_factory.py`, `registry.py`)**: Dynamic instantiation of trainer backends via `TrainerRegistry`.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `hf_trainer.py` | `HuggingFaceTrainer` wrapper integrating with `ResultsManager`. |
| `trainer.py` | `SVDAwareTrainer` supporting attention map logging and SVD callbacks. |
| `custom_loop_trainer.py` | High-performance PyTorch training loop with DDP synchronization safety. |
| `rl_trainer.py` | `RLTrainer` for preference optimization and policy-gradient fine-tuning. |
| `unsloth_trainer.py` | `UnslothTrainer` fast kernel trainer with transparent fallback. |
| `trainer_factory.py` | Factory function `get_trainer()` for resolving trainer classes. |
| `methods/` | Parameter-efficient fine-tuning methods (LoRA, GeLoRA, Mixout, LLRD, Fisher). |

---

## 🚀 Quick Start & Usage

### 1. Training with HuggingFace Trainer
```python
from framework.trainer.hf_trainer import HuggingFaceTrainer
from framework.trainer.results_manager import ResultsManager

results_mgr = ResultsManager(output_dir="./runs/experiment_01")
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config={"learning_rate": 2e-5, "epochs": 3, "batch_size": 32, "fp16": True},
    results_manager=results_mgr,
)
metrics = trainer.train()
```

### 2. High-Throughput Custom Loop Training
```python
from framework.trainer.custom_loop_trainer import CustomLoopTrainer

trainer = CustomLoopTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config={"learning_rate": 3e-5, "epochs": 5, "batch_size": 64, "amp": True},
)
trainer.train()
```
