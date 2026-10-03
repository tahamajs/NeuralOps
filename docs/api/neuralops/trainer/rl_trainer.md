Module neuralops.trainer.rl_trainer
===================================
RLTrainer – Reinforcement Learning and Preference Optimization Trainer.
Supports TRL (Transformer Reinforcement Learning: PPO, DPO, SFT) when available,
with a native PyTorch Policy-Gradient / Reward-Weighted Optimization fallback.

Classes
-------

`RLTrainer(model: nn.Module, train_dataset: Any = None, eval_dataset: Any = None, config: dict[str, Any] | None = None, results_manager: Any | None = None, reward_model: nn.Module | None = None, compute_metrics: Callable | None = None, **kwargs: Any)`
:   Reinforcement Learning & Alignment Trainer supporting Direct Preference Optimization (DPO),
    Proximal Policy Optimization (PPO), and native policy-gradient fine-tuning.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC

    ### Methods

    `evaluate(self) ‑> dict[str, typing.Any]`
    :   Evaluate policy performance.

    `train(self) ‑> dict[str, typing.Any]`
    :   Execute RL / policy fine-tuning loop.