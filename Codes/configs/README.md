
# Config Profiles – Pre‑built YAML Configurations

This folder contains **ready‑to‑use YAML configuration profiles** for the Efficient Transformer Toolkit CLI.  
They allow you to quickly switch between models, tasks, attention mechanisms, and training presets without typing long command lines.

All config files work with the `config_loader.py` module, which handles merging and precedence.  
They are also directly usable via the CLI with `--task`, `--model`, `--attention`, and `--preset` flags.

---

## 🗂️ Directory Structure

```
configs/
├── default.yaml                 # Base defaults (applied to all runs)
├── tasks/                       # Per‑task settings
│   ├── mrpc.yaml
│   ├── mnli.yaml
│   ├── sst2.yaml
│   ├── cola.yaml
│   └── qqp.yaml
├── models/                      # Per‑model settings
│   ├── distilbert_base.yaml
│   ├── bert_base.yaml
│   └── roberta_base.yaml
├── attentions/                  # Per‑attention settings
│   ├── full.yaml
│   ├── aurora.yaml
│   ├── nova.yaml
│   ├── prism_full.yaml
│   ├── minimal_nova.yaml
│   └── phantom.yaml
└── presets/                     # Combined presets (fast, accurate, extreme)
    ├── fast.yaml
    ├── accurate.yaml
    └── extreme_compress.yaml
```

---

## ⚙️ How Configs Work

### The config system builds a final configuration by merging multiple sources:

1. **`default.yaml`** – always loaded first; contains the base fallback values.
2. **Named profiles** – optional, loaded when using `--preset`, `--task`, `--model`, or `--attention`.
3. **Custom YAML files** – loaded via `--config my_exp.yaml` (can be repeated).
4. **Environment variables** – any field can be overridden with `ETK_<FIELD>` (e.g., `ETK_BATCH_SIZE=32`).
5. **CLI arguments** – highest precedence; override everything else.

**Precedence (highest to lowest):**  
CLI args → Environment variables → Custom YAML files → Named profiles → `default.yaml`

---

## 📄 Config File Format

All config files are plain YAML and use the same fields defined in the `TrainingConfig` dataclass.

### Example – a minimal attention config:

```yaml
# configs/attentions/aurora.yaml
attention_type: "aurora"
compression_rank: 64
config_description: "AURORA – 10‑in‑1 compression suite"
```

### Example – a preset config:

```yaml
# configs/presets/fast.yaml
model_name: "distilbert-base-uncased"
attention_type: "minimal_nova"
compression_rank: 32
batch_size: 64
max_seq_length: 64
fp16: true
config_description: "Fast preset – minimal compression, high throughput"
```

You only need to include the fields you want to override. All other fields remain at their default values.

---

## 🚀 Using Config Profiles

### From the CLI

- **Task profile:** `--task mrpc` → loads `configs/tasks/mrpc.yaml`
- **Model profile:** `--model bert_base` → loads `configs/models/bert_base.yaml`
- **Attention profile:** `--attention aurora` → loads `configs/attentions/aurora.yaml`
- **Preset profile:** `--preset fast` → loads `configs/presets/fast.yaml`
- **Custom YAML:** `--config my_experiment.yaml` → loads additional custom configs

You can combine them:

```bash
python cli.py train --task mnli --model roberta_base --attention aurora --preset accurate
```

The order of merging is: preset → task → model → attention → custom YAML(s).  
Later files override earlier ones for conflicting keys.

### Print resolved config

```bash
python cli.py train --task mrpc --attention nova --print-config
```

This prints the final merged configuration without starting training.

### Save resolved config for reproducibility

```bash
python cli.py train --task mrpc --attention aurora --save-config aurora_mrpc.yaml
```

### Interactive mode

```bash
python cli.py train --interactive
```

(needs `questionary` installed)

---

## 📋 Pre‑built Profiles

### Tasks (in `configs/tasks/`)

| File | dataset_name | num_labels | max_seq_length | Description |
|------|--------------|------------|----------------|-------------|
| `mrpc.yaml` | mrpc | 2 | 128 | Paraphrase detection |
| `mnli.yaml` | mnli | 3 | 128 | Natural language inference |
| `sst2.yaml` | sst2 | 2 | 64 | Sentiment analysis |
| `cola.yaml` | cola | 2 | 64 | Acceptability |
| `qqp.yaml` | qqp | 2 | 128 | Question pairs |

### Models (in `configs/models/`)

| File | model_name | Description |
|------|------------|-------------|
| `distilbert_base.yaml` | distilbert-base-uncased | 66M params |
| `bert_base.yaml` | bert-base-uncased | 110M params |
| `roberta_base.yaml` | roberta-base | 125M params |

### Attention (in `configs/attentions/`)

| File | attention_type | compression_rank | Description |
|------|----------------|------------------|-------------|
| `full.yaml` | full | 64 | Original attention |
| `aurora.yaml` | aurora | 64 | 10‑in‑1 compression |
| `nova.yaml` | nova | 64 | 8 techniques |
| `prism_full.yaml` | prism_full | 32 | 10‑20× reduction |
| `minimal_nova.yaml` | minimal_nova | 64 | SVD‑only zero‑shot |
| `phantom.yaml` | phantom | 16 | CUR + LSH |

### Presets (in `configs/presets/`)

| File | Combines | Goal |
|------|----------|------|
| `fast.yaml` | distilbert + minimal_nova + small batch | Speed, low memory |
| `accurate.yaml` | roberta + full attention + large seq | Maximum accuracy |
| `extreme_compress.yaml` | distilbert + prism_full + small rank | 20× parameter reduction |

---

## 🛠️ Creating Custom Configs

### 1. Scaffold a new config from defaults

```bash
python cli.py init-config -o path/to/my_config.yaml
```

This creates a YAML file with all fields set to their default values.

### 2. Start from a preset

```bash
python cli.py init-config --preset fast -o my_fast_config.yaml
```

### 3. Edit the file

Open `my_fast_config.yaml` and change any values. For example:

```yaml
model_name: "distilbert-base-uncased"
attention_type: "aurora"
compression_rank: 48
dataset_name: "mrpc"
batch_size: 32
num_epochs: 5
learning_rate: 3.0e-5
fp16: true
```

### 4. Use it

```bash
python cli.py train --config my_fast_config.yaml
```

Override specific fields from CLI if needed:

```bash
python cli.py train --config my_fast_config.yaml --epochs 10
```

---

## 🌐 Environment Variables

All config fields can be overridden via environment variables using the prefix `ETK_`.  
The variable name must be the **uppercase** version of the field name.

Examples:

```bash
export ETK_MODEL_NAME="bert-base-uncased"
export ETK_ATTENTION_TYPE="nova"
export ETK_NUM_EPOCHS=5
export ETK_BATCH_SIZE=64

python cli.py train --task mrpc   # will use the env var overrides
```

Useful in CI/CD pipelines or shell scripts.

---

## 📚 Reference – `TrainingConfig` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | str | `distilbert-base-uncased` | HuggingFace model identifier |
| `attention_type` | str | `full` | Registry key of attention module |
| `compression_rank` | int | 64 | Rank for low‑rank compression |
| `use_unsloth` | bool | false | Load with Unsloth |
| `cache_dir` | str? | null | HF cache directory |
| `dataset_name` | str | `mrpc` | Task name or custom file path |
| `dataset_path` | str? | null | Explicit path to dataset file |
| `max_seq_length` | int | 128 | Tokenization max length |
| `split_ratio` | float | 0.1 | Validation split for custom datasets |
| `glue_base_path` | str | `./data/glue` | Local GLUE Parquet path |
| `num_labels` | int? | null | Number of output labels (auto‑detected) |
| `output_dir` | str | `./output` | Model save directory |
| `num_epochs` | int | 3 | Training epochs |
| `batch_size` | int | 16 | Training batch size per device |
| `eval_batch_size` | int | 32 | Evaluation batch size |
| `learning_rate` | float | 2e-5 | Learning rate |
| `weight_decay` | float | 0.01 | Weight decay |
| `warmup_ratio` | float | 0.06 | Warmup steps ratio |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation |
| `max_grad_norm` | float | 1.0 | Gradient clipping |
| `seed` | int | 42 | Random seed |
| `device` | str | `auto` | Device (`cuda`, `cpu`, `auto`) |
| `fp16` | bool | true | Use mixed precision FP16 |
| `bf16` | bool | false | Use BFloat16 |
| `logging_steps` | int | 50 | Logging interval |
| `eval_steps` | int | 500 | Evaluation interval |
| `save_steps` | int | 500 | Checkpoint interval |
| `save_total_limit` | int | 2 | Max saved checkpoints |
| `compression_method` | str | `low_rank` | Compression method (for compress cmd) |
| `target_sparsity` | float | 0.5 | Sparsity for pruning |
| `target_rank` | int | 64 | Rank for compress cmd |
| `synthetic_domain` | str | `movies` | Domain for synthetic data |
| `synthetic_samples` | int | 5000 | Number of synthetic samples |
| `synthetic_output` | str | `./data/synthetic.csv` | Output path for synthetic data |
| `config_name` / `config_description` | str | – | Metadata (ignored by training) |

---

## 💡 Tips

- **Start with a preset** – they are the quickest way to get a sensible configuration.
- **Always save your config** with `--save-config` if you plan to share or reproduce the experiment.
- **Use `--print-config`** before a long run to verify everything is correct.
- **Combine multiple custom configs** by specifying `--config` multiple times.
- **Look at the existing profiles** for examples you can copy and modify.

---

*Config files make your experiments reproducible and shareable – use them wisely!*
