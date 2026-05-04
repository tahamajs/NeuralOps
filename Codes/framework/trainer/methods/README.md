# 🧪 Training Methods

A collection of parameter‑efficient fine‑tuning (PEFT) and regularisation methods for transformer models. All methods are registered in a central registry and can be applied via a unified interface.

## 📁 Module Structure

```
framework/trainer/methods/
├── __init__.py          # Exports registry and all methods
├── registry.py          # METHOD_REGISTRY and helpers
├── lora.py              # LoRA, DoRA, DisLoRA
├── fisher.py            # Fisher‑based adaptive LoRA
├── gelora.py            # GeLoRA (intrinsic dimension estimation)
├── llrd.py              # Layer‑wise Learning Rate Decay
├── mixout.py            # Mixout regularisation
├── rdrop_r3f.py         # R‑Drop and R3F regularisation
└── README.md            # This file
```

## 🔧 Registry

All methods are registered with a string key. Use the registry to apply a method by name:

```python
from framework.trainer.methods import get_method

apply_fn = get_method("lora")
apply_fn(model, config)   # modifies the model in‑place
```

List all available methods:

```python
from framework.trainer.methods import available_methods
print(available_methods())
```

## 🧬 Available Methods

### 1. `lora` – Low‑Rank Adaptation

**Paper**: [LoRA: Low‑Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

Replaces selected linear layers with `LoRALinear` modules. The pre‑trained weights are frozen, and two small matrices (`A` and `B`) are trained.

**Config parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lora_r` | Rank of the low‑rank matrices | 8 |
| `lora_alpha` | Scaling factor (`alpha / r`) | 16 |
| `lora_dropout` | Dropout probability | 0.1 |

**Example**:

```python
config.lora_r = 16
config.lora_alpha = 32
config.lora_dropout = 0.2
apply_lora(model, config)
```

---

### 2. `dora` – Weight‑Decomposed Low‑Rank Adaptation

**Paper**: [DoRA: Weight‑Decomposed Low‑Rank Adaptation](https://arxiv.org/abs/2402.09353)

Extends LoRA by learning a magnitude vector `m` that scales the direction of the base weights. Often yields better performance than standard LoRA.

**Config parameters**: Same as `lora` (`lora_r`, `lora_alpha`, `lora_dropout`).

**Example**: `config.method = "dora"` in your training config.

---

### 3. `dislora` – Dissimilarity‑based LoRA

Encourages orthogonality between the LoRA matrices to increase capacity and reduce redundancy.

**Additional config parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dislora_beta` | Weight for orthogonality regularisation | 0.5 |
| `dislora_capacity_factor` | Multiplier for rank capacity | 1.0 |

**Example**:

```python
config.lora_r = 8
config.dislora_beta = 0.3
apply_dislora(model, config)
```

---

### 4. `fisher_lora` – Fisher‑Information guided LoRA

Computes Fisher information on a small subset of the training data and assigns higher ranks to layers with higher Fisher scores. Adaptive ranks (rank per layer) are not yet fully implemented – the method currently computes and logs the Fisher scores.

**Config parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fisher_sample_size` | Number of samples for Fisher estimation | 200 |
| `fisher_num_batches` | Number of batches for estimation | 10 |
| `per_device_train_batch_size` | Batch size during Fisher estimation | 8 |
| `max_seq_length` | Max tokens per sample | 128 |

**Usage**: The method computes Fisher and logs per‑module importance; you can later use the scores to assign adaptive ranks.

---

### 5. `fisher_gelora` – Fisher + GeLoRA

Combines Fisher importance with **GeLoRA** (intrinsic dimension estimation). The intrinsic dimension of each layer’s hidden states is estimated, and LoRA ranks are set proportionally.

**GeLoRA steps**:

1. Hook into encoder layers to collect hidden states.
2. Compute intrinsic dimension using the nearest‑neighbour method.
3. Set base rank: `max(min_rank, min(max_rank, int(id + margin)))`.

**Config parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gelora_min_rank` | Minimum LoRA rank | 2 |
| `gelora_max_rank` | Maximum LoRA rank | 32 |
| `gelora_margin` | Additive margin to intrinsic dimension | 1.0 |
| `lora_r` | Fallback rank when layer cannot be resolved | 8 |
| `lora_alpha` | Scaling factor | 16 |
| `lora_dropout` | Dropout | 0.1 |

---

### 6. `llrd` – Layer‑wise Learning Rate Decay

**Paper**: [BERT‑LLRD: Fine‑tuning BERT with Layer‑wise Learning Rate Decay](https://arxiv.org/abs/1905.12616)

Applies different learning rates to different layers. Lower layers receive smaller learning rates, higher layers larger ones. The method does **not** modify the model; it returns parameter groups that can be passed to an optimizer.

**Config parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lr` | Base learning rate for the classifier | 2e‑5 |
| `weight_decay` | Weight decay for all groups | 0.01 |
| `llrd_decay` | Decay factor per layer (e.g., 0.95) | 0.95 |

**Usage**:

```python
from framework.trainer.methods import get_llrd_param_groups

optimizer_groups = get_llrd_param_groups(model, base_lr=2e-5, weight_decay=0.01, decay=0.95)
optimizer = AdamW(optimizer_groups)
```

---

### 7. `mixout` – Mixout Regularisation

**Paper**: [Mixout: Effective Regularization to Finetune Large‑scale Pretrained Language Models](https://arxiv.org/abs/1909.11299)

Replaces standard linear layers with `MixoutLinear`, which stochastically mixes the current weights with a fixed target weight (usually the pre‑trained weights) during training.

**Config parameter**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mixout_p` | Probability of using the current weight instead of the target | 0.5 |

**Example**: `config.mixout_p = 0.5`

---

### 8. `rdrop` – R‑Drop Regularisation

**Paper**: [R‑Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)

Trains the same mini‑batch twice with dropout active and adds a KL‑divergence term between the two forward passes.

**Config parameter**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rdrop_alpha` | Weight of the KL‑divergence loss | 5.0 |

**Integration**: The method does not modify the model; it expects the `MethodTrainer` (provided in `rdrop_r3f.py`) to be used. The training factory uses it automatically when `method="rdrop"`.

---

### 9. `r3f` – R3F (Randomised Re‑weighted Fisher)

**Paper**: [R3F: Randomised Re‑weighted Fisher for Robust Fine‑tuning](https://arxiv.org/abs/2109.12508)

Adds a small amount of noise to the input embeddings during training and enforces consistency (via KL divergence) between clean and noisy forward passes.

**Config parameter**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r3f_epsilon` | Standard deviation of the additive Gaussian noise | 1e‑5 |

**Integration**: Works via `MethodTrainer` when `method="r3f"`.

---

## 🚀 Integration with Training Factory

When you use `get_trainer` from `framework.training`, the `method` key in the config is automatically applied. For example:

```yaml
# config.yaml
method: "lora"
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
```

In your training script:

```python
from framework.training import get_trainer

config = load_config("config.yaml")
trainer, results_mgr = get_trainer(model, train_dataset, eval_dataset, config=config)
trainer.train()
```

The factory calls the appropriate `apply_*` function before starting training and also initialises the correct `MethodTrainer` for R‑Drop / R3F.

## 🔌 Adding a Custom Method

1. Create a new Python file in `framework/trainer/methods/`.
2. Define a function that takes `(model, config)` (and optionally `dataset`, `tokenizer`, `task_keys`) and modifies the model in‑place.
3. Decorate the function with `@register_method("your_method_name")`.

**Example**:

```python
from .registry import register_method

@register_method("my_method")
def apply_my_method(model, config):
    # Your adaptation logic here
    pass
```

The method will then be available via `get_method("my_method")`.

---

## 📚 References

- LoRA: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- DoRA: [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)
- GeLoRA: [Intrinsic Dimension Estimation](https://openreview.net/forum?id=zd81Y7JsgA)
- LLRD: [https://arxiv.org/abs/1905.12616](https://arxiv.org/abs/1905.12616)
- Mixout: [https://arxiv.org/abs/1909.11299](https://arxiv.org/abs/1909.11299)
- R‑Drop: [https://arxiv.org/abs/2106.14448](https://arxiv.org/abs/2106.14448)
- R3F: [https://arxiv.org/abs/2109.12508](https://arxiv.org/abs/2109.12508)

---

This module is part of the **Efficient Transformer Toolkit** – MIT License.
