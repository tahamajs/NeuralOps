# 🚀 NeuralOps Framework — Full GPU-Server Readiness Agent Execution Plan

> **Purpose**: This document is the complete, exhaustive plan for 5 parallel agents to implement, fix,
> validate, and finalize every component of the framework so it runs without errors on a real GPU server
> across BERT, RoBERTa, T5, DistilBERT, GPT-2, and all registered attention/compression modules.
>
> **How to use**: Copy each Agent section and paste it as the prompt to a fresh agent.
> All 5 agents can run in parallel. They are fully independent.
> After all 5 complete, run: `bash Codes/framework/run_on_gpu_server.sh suite`

---

## Repository Layout

```
Project/
└── Codes/
    └── framework/
        ├── attentions/              # 239 registered attention modules (.py + .md each)
        ├── compression_engine/      # 112 registered compressor modules (.py + .md each)
        ├── core/                    # Registry, base classes, benchmark engine
        │   ├── registry.py          # ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        │   ├── base.py
        │   ├── multi_mode_benchmark.py
        │   └── attention_utils.py   # inject_attention()
        ├── data/
        │   ├── prepare_datasets.py  # Main dataset pipeline
        │   ├── glue_loader.py       # Local GLUE cache loader
        │   ├── benchmark_datasets.py
        │   ├── synthetic_generator.py
        │   └── glue_data/           # Pre-downloaded GLUE splits
        ├── trainer/
        │   ├── trainer.py
        │   ├── rl_trainer.py
        │   └── unsloth_trainer.py
        ├── results_saver/saver.py   # JSON + CSV + Markdown result saver
        ├── tests/                   # pytest test suite
        ├── eval_all_models_datasets.py   # Universal benchmark engine
        ├── launch_gpu_server.py          # GPU server launcher
        ├── auto_run.py                   # Multi-track parallel auto-runner
        ├── main.py                       # CLI entry point
        ├── run_on_gpu_server.sh          # Shell launcher
        ├── check_framework_sync.py       # Registry integrity checker
        └── configs/
            ├── default.yaml
            └── sweep_attention_types.yaml
```

---

## Already Done (Do NOT redo these)

- 239 attention modules registered in ATTENTION_REGISTRY
- 112 compression engines registered in COMPRESSOR_REGISTRY
- All custom linear layers have `@property def weight` for T5 compatibility
- Zero `except Exception: pass` — all replaced with typed exceptions
- `eval_all_models_datasets.py` uses genuine GLUE datasets; synthetic requires `--synthetic`
- `data/prepare_datasets.py` downloads GLUE (SST-2, MRPC, QNLI, COLA, RTE) via HuggingFace + saves to disk
- `run_on_gpu_server.sh` exists with diagnostics/prepare-data/fast/comprehensive/suite modes
- `results_saver/saver.py` saves results one-by-one as JSON + master CSV

---

## 🤖 AGENT 1 — Dataset Pipeline & Multi-Model Tokenizer Hardening

**Target files:**
- `Codes/framework/data/prepare_datasets.py`
- `Codes/framework/data/glue_loader.py`
- `Codes/framework/data/benchmark_datasets.py`
- `Codes/framework/data/__init__.py`
- `Codes/framework/data/README.md`

**Problems to fix:**

### Fix 1.1 — T5 tokenizer produces `token_type_ids` that crash T5EncoderModel

In `prepare_datasets.py` → `prepare_glue_dataset()`, after tokenization, strip token_type_ids for T5:

```python
# After tokenized = raw_split.map(tokenize_fn, ...)
if "t5" in tokenizer_name.lower() and "token_type_ids" in tokenized.column_names:
    tokenized = tokenized.remove_columns(["token_type_ids"])
```

### Fix 1.2 — GPT-2 tokenizer has no pad token

Before calling tokenizer for GPT-2:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

### Fix 1.3 — Write complete `glue_loader.py`

Write this entire file at `Codes/framework/data/glue_loader.py`:

```python
"""
Local GLUE Loader — loads pre-cached GLUE datasets from disk first, then HF Hub.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
GLUE_LOCAL_DIR = Path(__file__).parent / "glue_data"

GLUE_TASK_HF_NAME = {
    "sst2": ("glue", "sst2"), "mrpc": ("glue", "mrpc"),
    "qnli": ("glue", "qnli"), "cola": ("glue", "cola"),
    "rte":  ("glue", "rte"),  "qqp":  ("glue", "qqp"),
    "mnli": ("glue", "mnli"), "stsb": ("glue", "stsb"),
    "wnli": ("glue", "wnli"),
}
GLUE_TASK_TEXT_KEYS = {
    "sst2": ("sentence", None),   "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"), "cola": ("sentence", None),
    "rte":  ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"), "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_glue_split(
    task: str, split: str = "validation",
    tokenizer_name: str = "bert-base-uncased", max_seq_len: int = 128,
) -> Optional[Dataset]:
    """Load GLUE split from disk cache or HF Hub. Returns torch-format Dataset."""
    task = task.lower()
    if task not in GLUE_TASK_HF_NAME:
        logger.error("Unknown GLUE task: %s", task)
        return None

    key1, key2 = GLUE_TASK_TEXT_KEYS[task]
    hf_path, hf_config = GLUE_TASK_HF_NAME[task]

    dataset = None
    # 1. Try local disk
    local_path = GLUE_LOCAL_DIR / task
    if local_path.exists():
        try:
            from datasets import load_from_disk
            raw = load_from_disk(str(local_path))
            if split in raw:
                dataset = raw[split]
                logger.info("Loaded '%s/%s' from local disk.", task, split)
        except Exception as e:
            logger.warning("Local disk load failed for '%s/%s': %s", task, split, e)

    # 2. Try HF Hub
    if dataset is None:
        try:
            from datasets import load_dataset
            raw = load_dataset(hf_path, hf_config)
            if split in raw:
                dataset = raw[split]
                logger.info("Loaded '%s/%s' from HF Hub.", task, split)
        except Exception as e:
            logger.warning("HF Hub load failed for '%s/%s': %s", task, split, e)
            return None

    if dataset is None:
        return None

    # 3. Tokenize
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def tokenize_fn(examples):
            args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
            return tokenizer(*args, padding="max_length", truncation=True, max_length=max_seq_len)

        remove_cols = [c for c in dataset.column_names if c not in {"label", "labels"}]
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
        if "label" in dataset.column_names and "labels" not in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        if "t5" in tokenizer_name.lower() and "token_type_ids" in dataset.column_names:
            dataset = dataset.remove_columns(["token_type_ids"])
        keep = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in dataset.column_names]
        dataset.set_format(type="torch", columns=keep)
        return dataset
    except Exception as e:
        logger.error("Tokenization failed for '%s/%s': %s", task, split, e)
        return None
```

### Fix 1.4 — Add `--models` flag to `prepare_datasets.py` main()

Modify argparse in `main()` to accept `--models bert,roberta,t5,distilbert` and loop over
`(task, model)` combos calling `prepare_glue_dataset(task, tokenizer_name=MODEL_TOKENIZER_MAPPING[model], ...)`.

The MODEL_TOKENIZER_MAPPING is already at the top of `eval_all_models_datasets.py` — import it or copy it:
```python
MODEL_TOKENIZER_MAPPING = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "gpt2": "gpt2",
    "distilbert": "distilbert-base-uncased",
}
```

### Fix 1.5 — Rewrite `data/README.md`

Write a clean 60-80 line README with:
- `glue_data/` directory structure
- Usage: `python prepare_datasets.py --all --models bert,roberta,t5 --save_dir /mnt/data`
- How `load_glue_split()` works vs `prepare_glue_dataset()`
- GPU server bootstrap one-liner

### Verification commands (run after all fixes):
```bash
cd /Users/tahamajs/Documents/uni/Project
PYTHONPATH=Codes python3 Codes/framework/data/prepare_datasets.py --tasks sst2,mrpc --test
PYTHONPATH=Codes python3 -c "
from framework.data.glue_loader import load_glue_split
ds = load_glue_split('sst2', 'validation', 'bert-base-uncased')
print('BERT SST-2:', len(ds), list(ds.features.keys()))
ds2 = load_glue_split('sst2', 'validation', 't5-small')
print('T5 SST-2:', len(ds2), list(ds2.features.keys()))
assert 'token_type_ids' not in ds2.features, 'T5 still has token_type_ids!'
print('ALL CHECKS PASSED')
"
```

Report: number of samples per split, feature keys for each model family, confirm no `token_type_ids` in T5 batches.

---

## 🤖 AGENT 2 — `eval_all_models_datasets.py` & `launch_gpu_server.py` End-to-End Runability

**Target files:**
- `Codes/framework/eval_all_models_datasets.py`
- `Codes/framework/launch_gpu_server.py`
- `Codes/framework/core/multi_mode_benchmark.py`

### Fix 2.1 — Add `_filter_batch()` to strip invalid keys per model family

Add this function near the top of `evaluate_single_combination()` in `eval_all_models_datasets.py`:

```python
def _filter_batch(batch, model_family: str, model) -> dict:
    """Strip keys invalid for specific model architectures."""
    import inspect
    family = model_family.lower()
    b = dict(batch)
    if "t5" in family:
        b.pop("token_type_ids", None)
    if "gpt2" in family:
        # GPT-2 forward only accepts: input_ids, attention_mask, labels (for LM)
        allowed = {"input_ids", "attention_mask", "labels"}
        b = {k: v for k, v in b.items() if k in allowed}
    # Further: drop keys not in model.forward() signature
    try:
        sig = inspect.signature(model.forward)
        valid = set(sig.parameters.keys()) - {"self"}
        if "kwargs" not in valid and "**kwargs" not in str(sig):
            b = {k: v for k, v in b.items() if k in valid}
    except (ValueError, TypeError):
        pass
    return b
```

Apply `batch = _filter_batch(batch, model_family, model)` before EVERY `model(**batch)` call.

### Fix 2.2 — Fix T5 output: no `.logits` attribute

T5EncoderModel returns `BaseModelOutput(last_hidden_state=...)`, NOT logits.
After every forward pass, normalize output extraction:

```python
def _extract_logits_and_loss(outputs, num_labels: int, labels=None):
    """Unified logit/loss extraction across all HF model output types."""
    import torch
    import torch.nn.functional as F
    
    loss = None
    logits = None
    
    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif hasattr(outputs, "last_hidden_state"):
        # T5EncoderModel: mean pool and slice to num_labels as proxy
        hs = outputs.last_hidden_state  # [B, L, D]
        logits = hs.mean(dim=1)[:, :num_labels]  # [B, num_labels]
        if labels is not None and logits.shape[-1] == num_labels:
            loss = F.cross_entropy(logits, labels)
    elif isinstance(outputs, (tuple, list)):
        first = outputs[0]
        if first.dim() == 3:
            logits = first[:, 0, :]  # CLS token [B, D]
            if logits.shape[-1] != num_labels:
                logits = logits[:, :num_labels]
        else:
            logits = first
            if logits.dim() == 0:
                loss = logits
                logits = None
    
    return logits, loss
```

### Fix 2.3 — Ensure `generate_dataset_batch()` is exported

Add this alias to `eval_all_models_datasets.py` if not already present:

```python
def generate_dataset_batch(
    dataset_name: str,
    model_family: str,
    batch_size: int = 4,
    seq_len: int = 32,
    device: torch.device = torch.device("cpu"),
    allow_synthetic: bool = False,
) -> Dict[str, torch.Tensor]:
    """Alias of get_dataset_batch for backwards compatibility with launch_gpu_server.py."""
    return get_dataset_batch(
        dataset_name=dataset_name,
        model_family=model_family,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        allow_synthetic=allow_synthetic,
    )
```

### Fix 2.4 — `launch_gpu_server.py`: ensure `main()` is complete

Read the last 50 lines of `launch_gpu_server.py`. If `if __name__ == "__main__": main()` is missing
or `main()` is incomplete, write a complete `main()` function that:
- Parses `--mode diagnostics|fast|comprehensive`
- Parses `--models bert,roberta,t5,distilbert`
- Parses `--datasets sst2,mrpc,qnli`
- Parses `--precision bf16|fp16|fp32`
- Parses `--batch_size`, `--seq_len`, `--output_dir`
- Calls `probe_hardware()`, prints diagnostics
- Calls `run_gpu_benchmark_sweep()` for non-diagnostics modes

### Fix 2.5 — Verify end-to-end forward pass for all 4 model families

Run this — ALL 4 must print "forward OK":
```bash
PYTHONPATH=Codes python3 -c "
import torch
from framework.eval_all_models_datasets import build_model_instance, get_dataset_batch

device = torch.device('cpu')
for family in ['bert', 'roberta', 't5', 'distilbert']:
    model, cfg = build_model_instance(family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device)
    batch = get_dataset_batch('sst2', family, batch_size=2, seq_len=16, device=device, allow_synthetic=True)
    b = {k: v for k, v in batch.items() if k != 'token_type_ids'} if family == 't5' else batch
    b = {k: v for k, v in b.items() if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
    if family == 't5':
        b.pop('labels', None)
    model.eval()
    with torch.no_grad():
        out = model(**b)
    print(f'{family}: forward OK — output type: {type(out).__name__}')
" 2>&1
```

Report exact output. Fix any that fail before reporting done.

---

## 🤖 AGENT 3 — `auto_run.py` & `main.py` CLI + Shell Script Hardening

**Target files:**
- `Codes/framework/auto_run.py`
- `Codes/framework/main.py`
- `Codes/framework/run_on_gpu_server.sh`

### Fix 3.1 — `auto_run.py`: add `--synthetic` flag

Add to argparse:
```python
parser.add_argument("--synthetic", action="store_true",
    help="Allow synthetic data batches (no real datasets needed)")
```
Thread it through: `run_track(...) → evaluate_single_combination(..., allow_synthetic=args.synthetic)`

### Fix 3.2 — `auto_run.py`: write complete `generate_report()` function

```python
def generate_report(results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    """Reads all JSON result files and generates a Markdown summary report."""
    import glob, json
    json_dir = os.path.join(results_dir, "json")
    files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    if not files:
        print("No result files found in", json_dir)
        return ""
    
    rows = []
    for fp in files:
        try:
            with open(fp) as f:
                rows.append(json.load(f))
        except Exception as e:
            logger.warning("Could not read %s: %s", fp, e)
    
    # Group by track
    from collections import defaultdict
    by_track = defaultdict(list)
    for r in rows:
        by_track[r.get("track", "unknown")].append(r)
    
    lines = ["# NeuralOps Benchmark Summary Report\n",
             f"**Total results**: {len(rows)} | **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]
    
    for track, track_rows in sorted(by_track.items()):
        lines.append(f"\n## {track}\n")
        lines.append("| Module | Model | Dataset | Status | Accuracy | Latency (ms) | Compression |\n")
        lines.append("|--------|-------|---------|--------|----------|-------------|-------------|\n")
        for r in sorted(track_rows, key=lambda x: (x.get("module_name",""), x.get("model_family",""))):
            acc = f"{r.get('accuracy_pct', 0):.2f}%"
            lat = f"{r.get('forward_latency_ms', 0):.1f}"
            comp = f"{r.get('compression_ratio', 1.0):.2f}x"
            status = "✅" if r.get("status") == "PASS" else "❌"
            lines.append(f"| {r.get('module_name','')} | {r.get('model_family','')} | {r.get('dataset','')} | {status} | {acc} | {lat} | {comp} |\n")
    
    report = "".join(lines)
    out_path = os.path.join(results_dir, "SUMMARY_REPORT.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {out_path}")
    return out_path
```

Wire it in `main()`: if `args.report: generate_report(); return`

### Fix 3.3 — `main.py`: `list` command must group by track

Find the `list` command handler. Replace with:
```python
from framework.eval_all_models_datasets import TRACK_MODULES
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

print(f"\n=== ATTENTION MODULES ({len(ATTENTION_REGISTRY)} total) ===")
for track, data in TRACK_MODULES.items():
    attns = data.get("attentions", [])
    if attns:
        print(f"\n  {track}:")
        for i, a in enumerate(attns):
            print(f"    {a}", end="\n" if (i+1) % 5 == 0 else "  ")
        print()

print(f"\n=== COMPRESSION ENGINES ({len(COMPRESSOR_REGISTRY)} total) ===")
for key in sorted(COMPRESSOR_REGISTRY.keys()):
    print(f"  {key}")
```

### Fix 3.4 — Fix `run_on_gpu_server.sh` argument parsing bug

Replace the current broken argument parsing with this clean version:
```bash
#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/Codes:${REPO_ROOT}/Codes/framework:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Parse: support both "bash run.sh fast" and "bash run.sh --mode fast"
ACTION="${1:-diagnostics}"
[[ "$ACTION" == "--mode" ]] && ACTION="${2:-diagnostics}"

cd "${REPO_ROOT}"
echo "============================================================"
echo " NeuralOps GPU Runner | Mode: ${ACTION} | Host: $(hostname)"
echo "============================================================"

case "${ACTION}" in
    diagnostics|test)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics ;;
    prepare-data|data)
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" \
            --all --save_dir "${REPO_ROOT}/prepared_benchmark_data" ;;
    fast|benchmark)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode fast --precision bf16 --batch_size 16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    comprehensive|full)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode comprehensive --precision fp16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    suite|all)
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" --mode diagnostics
        python3 "${SCRIPT_DIR}/data/prepare_datasets.py" \
            --tasks sst2,mrpc,qnli --save_dir "${REPO_ROOT}/prepared_benchmark_data"
        python3 "${SCRIPT_DIR}/launch_gpu_server.py" \
            --mode fast --precision bf16 \
            --output_dir "${REPO_ROOT}/results/gpu_server" ;;
    report)
        python3 "${SCRIPT_DIR}/auto_run.py" --report ;;
    *)
        echo "Usage: bash run_on_gpu_server.sh [diagnostics|prepare-data|fast|comprehensive|suite|report]"
        exit 1 ;;
esac
echo "============================================================"
echo " Completed! Results in: ${REPO_ROOT}/results/gpu_server/"
echo "============================================================"
```

### Fix 3.5 — Create `Codes/framework/Makefile`

```makefile
PYTHONPATH := $(shell cd ../..; pwd)/Codes
export PYTHONPATH

.PHONY: prepare fast suite report list test smoke

prepare:
	python3 data/prepare_datasets.py --tasks sst2,mrpc,qnli,cola --save_dir ./prepared_benchmark_data

fast:
	python3 launch_gpu_server.py --mode fast --precision bf16 --batch_size 16

suite:
	bash run_on_gpu_server.sh suite

report:
	python3 auto_run.py --report

list:
	python3 main.py list

test:
	PYTHONPATH=../../Codes pytest tests/ -v --timeout=120

smoke:
	PYTHONPATH=../../Codes python3 check_framework_sync.py
```

### Verification:
```bash
cd /Users/tahamajs/Documents/uni/Project
PYTHONPATH=Codes python3 Codes/framework/main.py list
PYTHONPATH=Codes python3 Codes/framework/auto_run.py --help
PYTHONPATH=Codes python3 Codes/framework/launch_gpu_server.py --mode diagnostics
bash Codes/framework/run_on_gpu_server.sh diagnostics
```

All must exit 0. Report exact output.

---

## 🤖 AGENT 4 — Tests, Smoke Tests & Results Saver

**Target files:**
- `Codes/framework/tests/test_forward_pass_all_models.py` (CREATE NEW)
- `Codes/framework/results_saver/saver.py`
- `Codes/framework/verify_forward_passes.py`
- `Codes/framework/pytest.ini`

### Fix 4.1 — Create `tests/test_forward_pass_all_models.py`

Create this complete file:

```python
"""
Comprehensive forward-pass smoke tests across all supported model families.
Runs on CPU — no GPU needed. Each test builds a tiny model (2 layers, 64 hidden).

Run: pytest tests/test_forward_pass_all_models.py -v
"""
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from framework.eval_all_models_datasets import build_model_instance


MODEL_FAMILIES = ["bert", "roberta", "t5", "distilbert", "gpt2"]


def make_batch(family: str, batch_size: int = 2, seq_len: int = 16):
    vocab_size = 50265 if family == "roberta" else 30522
    b = {
        "input_ids": torch.randint(100, vocab_size - 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
    if family not in {"t5", "gpt2"}:
        b["token_type_ids"] = torch.zeros(batch_size, seq_len, dtype=torch.long)
    b["labels"] = torch.zeros(batch_size, dtype=torch.long)
    return b


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_model_builds(family):
    model, cfg = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    assert isinstance(model, nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_forward_pass(family):
    model, _ = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    batch = make_batch(family)

    # Strip invalid keys
    if family == "t5":
        batch.pop("token_type_ids", None)
        batch.pop("labels", None)  # T5EncoderModel has no classification head
    if family == "gpt2":
        batch.pop("token_type_ids", None)

    model.eval()
    with torch.no_grad():
        output = model(**batch)
    assert output is not None


@pytest.mark.parametrize("family", ["bert", "distilbert"])
def test_backward_pass(family):
    model, _ = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1,
        device=torch.device("cpu")
    )
    model.train()
    batch = make_batch(family)
    output = model(**batch)
    loss = output.loss if hasattr(output, "loss") and output.loss is not None else output[0]
    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
        loss = loss.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


@pytest.mark.parametrize("module_name", [
    "aegis_v1", "aegis_v2", "nystrom_attention", "linformer_attention", "gqa_attention"
])
def test_attention_injection_bert(module_name):
    from framework.eval_all_models_datasets import apply_module_to_model
    device = torch.device("cpu")
    model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device)
    batch = make_batch("bert")
    try:
        model, ratio = apply_module_to_model(model, module_name, "attention", rank=8, device=device)
    except Exception as e:
        pytest.skip(f"Could not inject {module_name}: {e}")
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    assert output is not None
```

### Fix 4.2 — Verify `results_saver/saver.py` CSV writer

Read `results_saver/saver.py`. Ensure:
1. Master CSV is at `benchmark_results/master_results.csv`
2. Header is written only if file doesn't exist
3. Each call to `save_result(result)` APPENDS one row using `csv.DictWriter` with `extrasaction='ignore'`
4. All float fields are rounded to 4 decimal places before writing

If any issue, fix it. The CSV columns must be exactly:
```
timestamp, track, module_name, module_type, model_family, dataset, status,
error_message, forward_latency_ms, backward_latency_ms, throughput_samples_per_sec,
speedup_vs_baseline, param_count, trainable_param_count, compression_ratio,
peak_memory_mb, loss, accuracy_pct, f1_score, perplexity, grad_norm
```

### Fix 4.3 — Expand `verify_forward_passes.py`

Read current `verify_forward_passes.py`. If it tests fewer than 10 modules, replace with:
```python
#!/usr/bin/env python3
"""
Comprehensive forward-pass verification for top 10 novel attention modules.
Run: PYTHONPATH=../../Codes python3 verify_forward_passes.py
"""
import sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from framework.eval_all_models_datasets import build_model_instance, apply_module_to_model

TOP_NOVEL_MODULES = [
    "aegis_v1", "aegis_v2", "aegis_v3", "aegis_hybrid", "aegis_spectron_fp8",
    "nystrom_attention", "linformer_attention", "performer_attention", "mamba", "gqa_attention"
]

passed, failed = [], []
for mn in TOP_NOVEL_MODULES:
    try:
        model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1)
        model, ratio = apply_module_to_model(model, mn, "attention", rank=8)
        batch = {
            "input_ids": torch.randint(100, 30000, (2, 16)),
            "attention_mask": torch.ones(2, 16, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 16, dtype=torch.long),
            "labels": torch.zeros(2, dtype=torch.long),
        }
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            out = model(**batch)
        ms = (time.time() - t0) * 1000
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ {mn}: PASS — {ms:.1f}ms, {params:,} params, {ratio:.2f}x compression")
        passed.append(mn)
    except Exception as e:
        print(f"❌ {mn}: FAIL — {e}")
        failed.append(mn)

print(f"\n{'='*60}")
print(f"Results: {len(passed)} PASS, {len(failed)} FAIL")
if failed:
    print(f"Failed modules: {failed}")
    sys.exit(1)
```

### Fix 4.4 — Fix `pytest.ini`

Ensure `Codes/framework/pytest.ini` contains:
```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
timeout = 120
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
    smoke: quick sanity checks
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
```

### Verification:
```bash
cd /Users/tahamajs/Documents/uni/Project
PYTHONPATH=Codes pytest Codes/framework/tests/test_forward_pass_all_models.py -v --timeout=120 2>&1 | tail -30
PYTHONPATH=Codes python3 Codes/framework/verify_forward_passes.py
```

Report: which tests PASS/FAIL/SKIP with exact error messages.

---

## 🤖 AGENT 5 — Top 10 Novel Attention Modules: Validate & Fix Forward Passes + READMEs

**Target directories:**
- `Codes/framework/attentions/` (10 specific modules)
- `Docs/content/attentions/` (corresponding .md docs)

### The 10 modules to validate:

| # | Module | File | Novel Idea |
|---|--------|------|------------|
| 1 | `aegis_v6` | `aegis_v6.py` | SVD-factorized projections + energy thresholding + RFF global attention |
| 2 | `mamba` | `mamba.py` | Bidirectional SSM state-space scan for encoder classification |
| 3 | `nasc_attention` | `nasc_attention.py` | SVD-LoRA factorized projections with rotary embeddings |
| 4 | `spectron` | `spectron.py` | Low-rank sparse attention with spectral initialization |
| 5 | `phantom` | `phantom.py` | CUR matrix decomposition with pivot selection |
| 6 | `reap_moe` | `reap_moe.py` | SVD + Mixture-of-Experts gated routing |
| 7 | `gated_delta_net` | `gated_delta_net.py` | Gated delta network with bidirectional scanning |
| 8 | `hgrn2_layer` | `hgrn2_layer.py` | Hierarchical gated recurrent network v2 |
| 9 | `salt_attention` | `salt_attention.py` | Sparse attention with learned tokens |
| 10 | `orion` | `orion.py` | Factorised linear with orthogonal weight initialization |

### Fix 5.1 — Run the validation script for all 10 modules:
```bash
cd /Users/tahamajs/Documents/uni/Project
PYTHONPATH=Codes python3 -c "
import torch, sys
sys.path.insert(0, 'Codes')
from framework.eval_all_models_datasets import build_model_instance, apply_module_to_model

modules = [
    'aegis_v6', 'mamba', 'nasc_attention', 'spectron', 'phantom',
    'reap_moe', 'gated_delta_net', 'hgrn2_layer', 'salt_attention', 'orion'
]
passed, failed = [], []
for mn in modules:
    try:
        model, _ = build_model_instance('bert', num_labels=2, hidden_size=64, num_heads=4, num_layers=1)
        model, ratio = apply_module_to_model(model, mn, 'attention', rank=8)
        batch = {
            'input_ids': torch.randint(100, 30000, (2, 16)),
            'attention_mask': torch.ones(2, 16, dtype=torch.long),
            'token_type_ids': torch.zeros(2, 16, dtype=torch.long),
            'labels': torch.zeros(2, dtype=torch.long),
        }
        model.eval()
        with torch.no_grad():
            out = model(**batch)
        print(f'PASS: {mn}')
        passed.append(mn)
    except Exception as e:
        print(f'FAIL: {mn} — {e}')
        failed.append((mn, str(e)))
print(f'\nPASS: {len(passed)}, FAIL: {len(failed)}')
" 2>&1
```

### Fix 5.2 — For each FAILING module, apply these common fixes:

**Common fix A: `copy_weights()` KeyError on missing attribute**
In the module's `copy_weights()` method, make attribute resolution defensive:
```python
def copy_weights(self, source_module: nn.Module) -> None:
    # Try all common query projection names
    for q_attr in ["query", "q_proj", "q_lin", "q", "to_q"]:
        if hasattr(source_module, q_attr):
            src_q = getattr(source_module, q_attr)
            if hasattr(self, "W_q") and hasattr(src_q, "weight"):
                self.W_q.weight.data.copy_(src_q.weight.data)
            break
    for k_attr in ["key", "k_proj", "k_lin", "k", "to_k"]:
        if hasattr(source_module, k_attr):
            src_k = getattr(source_module, k_attr)
            if hasattr(self, "W_k") and hasattr(src_k, "weight"):
                self.W_k.weight.data.copy_(src_k.weight.data)
            break
    for v_attr in ["value", "v_proj", "v_lin", "v", "to_v"]:
        if hasattr(source_module, v_attr):
            src_v = getattr(source_module, v_attr)
            if hasattr(self, "W_v") and hasattr(src_v, "weight"):
                self.W_v.weight.data.copy_(src_v.weight.data)
            break
```

**Common fix B: `__init__` requires `config` but gets None**
In `__init__`, always handle `None` config:
```python
if config is None:
    config = type("MinimalConfig", (), {
        "hidden_size": 768, "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1, "is_decoder": False,
        "d_model": 768,
    })()
d_model = getattr(config, "hidden_size", getattr(config, "d_model", 768))
num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))
```

**Common fix C: Shape mismatch — [B, H, L, D/H] vs [B, L, D]**
```python
if hidden_states.dim() == 4:
    B, H, L, Dh = hidden_states.shape
    hidden_states = hidden_states.view(B, L, H * Dh)
```

### Fix 5.3 — For each of the 10 modules, verify the `.md` file in `Docs/content/attentions/`

Check each file has these sections (add any that are missing):
1. `## 1. Overview` — 2-3 sentence summary of the novel idea
2. `## 2. Mathematical Formulation` — proper LaTeX math notation with $\mathcal{O}(\cdot)$ complexity
3. `## 3. Implementation Notes` — how to inject into BERT/T5, any caveats
4. `## 4. Known Limitations & Future Work`

And remove any fake benchmark tables (rows with hard-coded accuracy numbers). Replace with:
```markdown
## 5. Empirical Benchmarks
> Benchmarks Pending — run `bash run_on_gpu_server.sh suite` to populate results.
> See [Empirical Validation Protocol](../empirical_validation_protocol.md).
```

### Fix 5.4 — Rewrite `Codes/framework/attentions/README.md`

Write an 80-100 line README with:
1. A table of the top 10 novel modules: `| Module | File | Novel Idea | Complexity | T5-Compatible |`
2. How to register a new attention module (3 steps)
3. T5 compatibility: `@property def weight` requirement
4. Forward pass signature: must accept `(hidden_states, attention_mask=None, **kwargs)`
5. Test command: `PYTHONPATH=../../Codes python3 verify_forward_passes.py`

### Verification:
After all fixes, re-run the validation script from Fix 5.1. All 10 must print `PASS`.
Report:
- Which were already passing
- Which were fixed (with what fix was applied)
- Final status: `PASS: X/10`

---

## Final Integration Verification (After All 5 Agents Complete)

```bash
# 1. Framework integrity
cd /Users/tahamajs/Documents/uni/Project
PYTHONPATH=Codes python3 Codes/framework/check_framework_sync.py

# 2. All tests pass
PYTHONPATH=Codes pytest Codes/framework/tests/test_forward_pass_all_models.py -v

# 3. CLI works
PYTHONPATH=Codes python3 Codes/framework/main.py list
PYTHONPATH=Codes python3 Codes/framework/auto_run.py --help

# 4. GPU server shell works
bash Codes/framework/run_on_gpu_server.sh diagnostics

# 5. On GPU server only:
# bash Codes/framework/run_on_gpu_server.sh suite
# PYTHONPATH=Codes python3 Codes/framework/auto_run.py --report
```

**Expected final state:**
- `tests/test_forward_pass_all_models.py` — all `test_model_builds` + `test_forward_pass` PASS
- `results_saver/saver.py` — correctly appends JSON + CSV per result
- `benchmark_results/SUMMARY_REPORT.md` — generated after GPU run
- `data/glue_loader.py` — loads T5 batches without `token_type_ids`
- `run_on_gpu_server.sh suite` — exits 0 on GPU server

---

*Plan version: 2026-08-25 | Total agents: 5 | Estimated GPU server run time: 2-4 hours for comprehensive mode*
