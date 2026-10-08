# NeuralOps API Reference (v1.0.0)

Exhaustive reference for core public interfaces across NeuralOps subsystems.

---

## 1. Core Registry (`neuralops.core.registry`)

### Classes & Constants
- `Registry(name: str)`: Central lazy/eager registry container with alias resolution and deprecation tracking.
  - `register(name: str, aliases: list[str] | None = None)`: Decorator or method to register module classes/factories.
  - `get(name: str) -> Any`: Retrieve registered module; emits deprecation warning if alias is used.
  - `keys() -> list[str]`: Returns canonical registered keys.
  - `all_keys() -> list[str]`: Returns both canonical keys and aliases.
- `ATTENTION_REGISTRY`: Canonical registry for all 159 attention mechanisms.
- `COMPRESSOR_REGISTRY`: Canonical registry for all 73 model compressor modules.

---

## 2. Optimizers & Schedulers (`neuralops.trainer.optimizers`, `neuralops.trainer.schedulers`)

### Optimizers
- `create_adamw(params, lr=1e-3, weight_decay=0.01)`
- `create_sgd(params, lr=0.01, momentum=0.9, weight_decay=0.0)`
- `create_adam(params, lr=1e-3, weight_decay=0.0)`
- `create_muon(params, lr=0.02, momentum=0.95, weight_decay=0.01)`
- `create_muon_adamw_reg(model, lr=0.02, lr_adam=5e-5, weight_decay=0.01)`
- `create_riemannian_stiefel(params, lr=1e-3, lr_stiefel=None, base_optimizer="adamw")`
- `create_adabelief(params, lr=1e-3, weight_decay=0.0)`
- `create_ranger(params, lr=1e-3, weight_decay=0.0)`
- `create_lamb(params, lr=1e-3, weight_decay=0.01)`
- `create_lars(params, lr=1e-3, weight_decay=0.0)`

### Schedulers
- `create_linear(optimizer, num_warmup_steps=0, num_training_steps=10000)`
- `create_cosine(optimizer, T_max=10000)`
- `create_rop(optimizer, factor=0.1, patience=10)`
- `create_onecycle(optimizer, max_lr=0.01, steps_per_epoch=1000, epochs=10)`
- `create_cosine_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)`
- `create_poly(optimizer, total_iters=10000, power=1.0)`

---

## 3. Data Infrastructure (`neuralops.data`)

### Base Data Types & Factories (`neuralops.data.base`)
- `DataConfig`: Dataclass specifying dataset name, split, batch size, seq_len, tokenization rules.
- `TaskType`: Enum denoting classification, generation, regression, multi_choice.
- `create_dataloader(dataset, batch_size, shuffle, collator)`: Production PyTorch DataLoader factory.

### Cached Tensors (`neuralops.data.cached_tensors`)
- `get_cached_tokenized_dataset(dataset_name, model_family, seq_len, cache_dir)`: Loads or caches pre-tokenized tensors in RAM and disk.

### Synthetic Generators (`neuralops.data.synthetic_generator`)
- `generate_zipfian_tokens(num_samples, seq_len, vocab_size, alpha, seed)`: Generates text tokens matching Zipfian distributions.
- `generate_zipfian_markov_tokens(num_samples, seq_len, vocab_size, alpha, transition_mix, seed)`: Generates Markovian token sequences.

---

## 4. Benchmark CLI (`neuralops.benchmarks.cli`)

- Command interface: `python -m neuralops.benchmarks {run,resume,plan,list,verify,results} [args]`
