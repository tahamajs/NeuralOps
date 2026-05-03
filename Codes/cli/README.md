# 🧰 CLI – Efficient Transformer Toolkit

The command‑line interface (CLI) provides a quick and flexible way to **train**, **evaluate**, **compress**, and **generate synthetic data** using the Efficient Transformer Toolkit.  
All commands are powered by the same modular framework, making it easy to reproduce experiments, benchmark different attention mechanisms, or scale to production.

## 📦 Requirements

- Python 3.9+
- All dependencies listed in the project’s `requirements.txt`
- The `framework` package accessible (place `cli.py` inside `Codes/cli/` as described)

Install extra dependencies if you plan to use Unsloth:

```bash
pip install unsloth
```

## 🚀 Quick Start

```bash
# Show available commands
python cli.py --help

# Train DistilBERT with AURORA attention on MRPC
python cli.py train -m distilbert-base-uncased -t mrpc -a aurora -r 64 -o my_model

# Evaluate the trained model
python cli.py evaluate --model_path my_model -t mrpc

# Compress a fresh model
python cli.py compress -m distilbert-base-uncased --method low_rank --target-rank 32 -o compressed

# Generate synthetic data
python cli.py generate-synthetic -o data/synthetic.csv --samples 10000 --domain movies

# List all registered attention types
python cli.py list-attentions
```

## ⚙️ Commands

### 1. `train`

Train a model on a dataset with a specified attention mechanism.

```
Usage: cli.py train [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | `distilbert-base-uncased` | HuggingFace model name or path |
| `--attention` | `-a` | `full` | Registered attention key (e.g., `aurora`, `nova`, `prism_full`) |
| `--rank` | `-r` | `64` | Compression rank (passed to the attention injector) |
| `--device` | `-d` | `cuda` (if available) | Torch device |
| `--use-unsloth` | | `False` | Load with Unsloth (4‑bit) for supported models |
| `--cache-dir` | | `None` | HuggingFace cache directory |
| `--task` | `-t` | `mrpc` | GLUE task name or path to a custom CSV/JSON/Parquet file |
| `--max-seq-length` | | `128` | Maximum tokenization length |
| `--split-ratio` | | `0.1` | Validation split when no validation set exists |
| `--glue-path` | | `./data/glue` | Path to local GLUE Parquet files |
| `--output` | `-o` | `./output` | Directory to save the trained model and tokenizer |
| `--epochs` | | `3` | Number of training epochs |
| `--batch-size` | | `16` | Per‑device training batch size |
| `--lr` | | `2e-5` | Learning rate |
| `--num-labels` | | auto‑detected | Number of output labels (overrides automatic detection) |

**Example:**

```bash
python cli.py train -m bert-base-uncased -t qqp -a minimal_nova -r 32 --batch-size 32 --epochs 5
```

The script automatically:
- Loads the model and injects the chosen attention.
- Builds the dataset (GLUE or custom).
- Tokenizes and prepares the data.
- Initialises a HuggingFace `Trainer` and runs training.
- Saves the final model and tokenizer to `--output`.

### 2. `evaluate`

Evaluate a trained model on a GLUE task.

```
Usage: cli.py evaluate [OPTIONS]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--model_path` | ✅ | – | Path to the saved model directory |
| `--task` | | `mrpc` | GLUE task name |
| `--glue-path` | | `./data/glue` | Local GLUE data path |
| `--max-seq-length` | | `128` | Tokenization length |
| `--batch-size` | | `16` | Evaluation batch size |
| `--device` | | `cuda` (if available) | Device |

**Example:**

```bash
python cli.py evaluate --model_path ./output --task mnli
```

This loads the model from disk, runs evaluation on the validation/test split, and prints the results.

### 3. `compress`

Compress a model using a compression method from the registry.

```
Usage: cli.py compress [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `distilbert-base-uncased` | Model to compress |
| `--method` | `low_rank` | Compression key registered in `COMPRESSOR_REGISTRY` |
| `--target-sparsity` | `0.5` | Sparsity ratio (for pruning / hybrid) |
| `--target-rank` | `64` | Rank for low‑rank / hybrid methods |
| `--output` | `./compressed_model` | Output directory |
| `--device` | `cuda` (if available) | Device |

**Example:**

```bash
python cli.py compress -m roberta-base --method hybrid --target-sparsity 0.4 --target-rank 32 -o roberta_compressed
```

The compressor replaces all Linear layers with the compressed variant and saves the model.

### 4. `generate-synthetic`

Generate a synthetic classification dataset for quick prototyping.

```
Usage: cli.py generate-synthetic [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `data/synthetic.csv` | Path to save the CSV |
| `--samples` | `5000` | Total number of samples (balanced) |
| `--domain` | `movies` | Domain (`movies`, `tech`, `restaurants`, `books`) |
| `--seed` | `42` | Random seed |

**Example:**

```bash
python cli.py generate-synthetic -o data/my_data.csv --samples 20000 --domain tech
```

### 5. `list-attentions`

Print all attention types registered in the framework.

```bash
python cli.py list-attentions
```

## 📁 Project Structure

```
Codes/cli/
├── __init__.py      # (empty)
├── cli.py           # Main CLI script
└── README.md        # This file
```

The CLI imports from the main `framework` package located in the parent directory (adjusted via `sys.path`).

## 🔧 Extending the CLI

### Adding a new command

1. Define a new function in `cli.py` decorated with `@cli.command()`.
2. Add click options as needed.
3. Use existing framework components (loading, training, compression, data).
4. The help text is automatically generated.

Example skeleton:

```python
@cli.command()
@click.option('--param')
def my_new_command(param):
    """Do something useful."""
    click.echo(f"Running with {param}")
```

### Using custom data loaders

The `train` and `evaluate` commands use `framework.data.dataset_loader.load_dataset()` which reads an `ExperimentConfig`. You can modify that function to support additional dataset types (e.g., from HuggingFace Hub).

---

## 💡 Pro Tips

- **Use `--help`** with any command to see all available options.
- **Mix and match** attention types – any registered key works, including your own custom modules.
- **Batch size** and **sequence length** directly impact GPU memory; adjust them for your hardware.
- **Unsloth** works best with Llama, Mistral, and similar architectures; add `--use-unsloth` to load them efficiently.

---

*Happy building with the Efficient Transformer Toolkit!*
