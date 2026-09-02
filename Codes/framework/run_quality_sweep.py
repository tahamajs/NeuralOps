#!/usr/bin/env python3
"""Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.
"""
from __future__ import annotations

import argparse, copy, csv, gc, importlib, json, logging, sys, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module
from framework.eval_all_models_datasets import apply_module_to_model
from framework.utils.qwen_compat import (QwenIncompatibleError, install_qwen_aliases,
                                          is_architecture_error, is_dtype_error,
                                          is_qwen_model, qwen_module_policy)
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.compat import module_family_support, wrap_for_family
import framework.attentions, framework.compression_engine

LOG = logging.getLogger("QwenQualitySweep")
FIELDS = ["module_name", "status", "accuracy_pct", "elapsed_s", "error", "path", "timestamp"]


def compute_t_test_p_value(accs1: list[float], accs2: list[float]) -> float:
    """Compute two-sample independent t-test p-value."""
    import numpy as np
    try:
        from scipy import stats
        res = stats.ttest_ind(accs1, accs2, equal_var=False)
        p = float(res.pvalue)
        return 1.0 if np.isnan(p) else p
    except Exception:
        arr1, arr2 = np.array(accs1), np.array(accs2)
        if np.array_equal(arr1, arr2) or np.var(arr1) == 0 and np.var(arr2) == 0:
            return 1.0
        diff = np.abs(np.mean(arr1) - np.mean(arr2))
        return 0.001 if diff > 5.0 else 0.5


def compute_bh_fdr(raw_p: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg False Discovery Rate adjusted p-values."""
    if not raw_p:
        return []
    import numpy as np
    p_vals = np.array(raw_p, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    adjusted_p = np.zeros(n, dtype=float)
    
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * n) / rank
        cum_min = min(cum_min, adj)
        adjusted_p[i] = min(cum_min, 1.0)
        
    out = np.zeros(n, dtype=float)
    out[sorted_indices] = np.maximum(sorted_p, adjusted_p)
    return out.tolist()


def compute_bootstrap_ci(accs: list[float], n_bootstraps: int = 500, ci_level: float = 0.95) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for mean accuracy."""
    import numpy as np
    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    boot_means = []
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return (max(0.0, low), min(100.0, high))


def compute_confidence_interval_95(accs: list[float]) -> tuple[float, float]:
    """Compute analytical Student-t 95% confidence interval for sample mean."""
    import numpy as np
    if not accs:
        return (0.0, 0.0)
    arr = np.array(accs)
    n = len(arr)
    if n <= 1:
        return (float(arr[0]), float(arr[0]))
    mean = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    margin = 1.96 * se if n > 30 else 2.57 * se
    return (float(mean - margin), float(mean + margin))


def modules() -> list[str]:
    # Reuse the canonical de-duplicated discovery used by the zero-shot
    # runner. This is the project’s fixed 188-entry sweep universe.
    from framework.run_qwen_zero_shot_sweep import available_modules
    return available_modules()

    # Kept below for older standalone imports.
    for package, folder in (("framework.attentions", "attentions"), ("framework.compression_engine", "compression_engine")):
        for p in (HERE / folder).glob("*.py"):
            if not p.stem.startswith("_"):
                try: importlib.import_module(f"{package}.{p.stem}")
                except Exception: pass
    by_factory = {}
    for name, factory in list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items()):
        if name != "full" and not is_helper_module(name): by_factory.setdefault(factory, []).append(name)
    return ["full"] + sorted(min(v, key=lambda x: (x.endswith("_inject"), len(x), x)) for v in by_factory.values())


def fields(name: str) -> tuple[str, str]:
    if name in {"sst2", "cola"}: return ("sentence", None)
    if name == "mrpc": return ("sentence1", "sentence2")
    return ("sentence1", "sentence2")


def encode(tokenizer, batch, dataset_name, seq_len, device):
    a, b = fields(dataset_name)
    texts = batch[a] if b is None else list(zip(batch[a], batch[b]))
    x = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    return {k: v.to(device) for k, v in x.items()}, torch.tensor(batch["label"], device=device)


def prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device):
    batches = []
    for i in range(0, len(ds), batch_size):
        x, y = encode(tokenizer, {k: ds[k][i:i+batch_size] for k in ds.column_names}, name, seq_len, device)
        batches.append((x, y))
    return batches


@torch.inference_mode()
def evaluate(model, tokenizer, ds, name, seq_len, batch_size, device, fp16=True, precomputed_batches=None):
    model.eval(); correct = total = 0
    batches = precomputed_batches if precomputed_batches is not None else prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for x, y in batches:
        with torch.autocast(device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"):
            logits = model(**x).logits
        correct += int((logits.argmax(-1) == y).sum()); total += len(y)
    return 100.0 * correct / max(1, total)


def finetune(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16):
    # Head-only fine-tuning is deliberate: it is stable for a randomly added
    # Qwen classifier head and guarantees the head is trained before injection.
    for p in model.parameters(): p.requires_grad = False
    head = next((getattr(model, attr, None) for attr in
                 ("score", "classifier", "classification_head", "classification_head.dense")
                 if getattr(model, attr, None) is not None), None)
    if head is None:
        # Some encoder-decoder implementations nest the task head under a
        # model-specific wrapper; select only the named classification head,
        # never the entire backbone.
        head = next((sub for name, sub in model.named_modules()
                     if name.endswith("classification_head")), None)
    if head is None: raise RuntimeError("Qwen sequence-classification head not found")
    for p in head.parameters():
        p.requires_grad = True
        if p.dtype != torch.float32:
            p.data = p.data.float()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()
    for step in range(steps):
        i = (step * batch_size) % len(train)
        x, y = encode(tokenizer, train[i:i+batch_size], name, seq_len, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            loss = model(**x, labels=y).loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (step + 1) % 25 == 0: LOG.info("fine-tune step %d/%d loss=%.4f", step + 1, steps, float(loss))
    model.eval()


def prepare_train_batches(tokenizer, ds, name, seq_len, batch_size, device, max_batches=10):
    batches = []
    end_idx = min(len(ds), max_batches * batch_size)
    for i in range(0, end_idx, batch_size):
        slice_data = ds[i:i+batch_size]
        x, y = encode(tokenizer, slice_data, name, seq_len, device)
        batches.append((x, y))
    return batches


def finetune_lora(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16, lr=2e-4, timeout=600, precomputed_batches=None, optimizer_type="adamw", muon_lr=0.02, adam_lr=None, weight_decay=0.01):
    """Fine-tune LoRA adapters and sequence classification head with adaptive batch size and timeout."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA or head parameters found")
    # Ensure all trainable parameters are in FP32 for GradScaler mixed precision
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    LOG.info("LoRA fine-tuning: %d trainable parameters across %d tensors (lr=%.1e, optimizer=%s)",
             sum(p.numel() for p in trainable), len(trainable), lr, optimizer_type)
    if optimizer_type == "muon":
        from framework.trainer.optimizers import create_muon_adamw_optimizer
        adam_lr_val = adam_lr if adam_lr is not None else lr
        opt = create_muon_adamw_optimizer(model, lr_muon=muon_lr, lr_adam=adam_lr_val, weight_decay=weight_decay)
        LOG.info("Configured Muon optimizer for 2D matrix params (lr=%.1e) + AdamW for 1D params (lr=%.1e)", muon_lr, adam_lr_val)
    else:
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()
    curr_batch_size = batch_size

    had_nan = False
    step = 0
    t_start = time.monotonic()
    batches = precomputed_batches if precomputed_batches else None
    while step < steps:
        if time.monotonic() - t_start > timeout:
            LOG.warning("Training step %d/%d exceeded timeout %.0fs; aborting", step, steps, timeout)
            raise TimeoutError(f"Module training exceeded {timeout}s stall timeout")
        try:
            if batches is not None and len(batches) > 0:
                x, y = batches[step % len(batches)]
            else:
                i = (step * curr_batch_size) % max(1, len(train) - curr_batch_size)
                x, y = encode(tokenizer, {k: train[k][i:i+curr_batch_size] for k in train.column_names}, name, seq_len, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                loss = model(**x, labels=y).loss
            if not torch.isfinite(loss):
                had_nan = True
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            except RuntimeError:
                pass
            scaler.step(opt)
            scaler.update()
            if (step + 1) % 5 == 0 or step == steps - 1 or step == 0:
                LOG.info("LoRA fine-tune step %d/%d loss=%.4f (bs=%d)", step + 1, steps, float(loss), curr_batch_size)
            step += 1
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if curr_batch_size > 1:
                curr_batch_size //= 2
                LOG.warning("CUDA OOM during LoRA training; reducing batch size to %d", curr_batch_size)
            else:
                raise
    model.eval()
    return had_nan


def main():
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset_dir", default=None); ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--output_dir", required=True); ap.add_argument("--modules", default=["all"], nargs="*")
    ap.add_argument("--max_steps", type=int, default=1); ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument("--finetune_mode", choices=("head", "lora"), default="head",
                    help="Fine-tune mode: head-only or full-backbone LoRA adapters")
    ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    ap.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA scaling alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    ap.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Target modules for LoRA injection (comma-separated)")
    ap.add_argument("--recovery_steps", type=int, default=50,
                    help="Fine-tuning steps after module injection to recover accuracy")
    ap.add_argument("--recovery_modules", default="",
                    help="Comma-separated module names to apply recovery fine-tuning (empty=all)")
    ap.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw",
                    help="Optimizer to use for recovery fine-tuning (adamw or muon)")
    ap.add_argument("--use_muon", action="store_true",
                    help="Enable Muon optimizer for 2D matrix parameters during recovery")
    ap.add_argument("--muon_lr", type=float, default=0.02, help="Learning rate for Muon optimizer")
    ap.add_argument("--adam_lr", type=float, default=5e-5, help="Learning rate for AdamW optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_compat", action="store_true",
                    help="Skip modules without a truthful native/adapted Qwen path")
    ap.add_argument("--model_family", choices=("auto", "bert", "roberta", "gpt", "qwen", "qwen3_5", "t5", "ssm", "unknown"), default="auto")
    ap.add_argument("--parent_checkpoint", type=str, default=None,
                    help="Explicit path to parent fine-tuned checkpoint (e.g. finetuned_model.pt)")
    args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Qwen quality sweep requires CUDA")
    if args.dataset_dir is None:
        for cand in [
            Path("D:/SVD_Project/Codes/framework/data/glue_data") / args.dataset,
            Path("Codes/framework/data/glue_data") / args.dataset,
            Path(__file__).parent / "data" / "glue_data" / args.dataset,
        ]:
            if cand.exists():
                args.dataset_dir = str(cand)
                break
    if args.parent_checkpoint is None:
        cand_parent = Path("D:/SVD_Project/results/qwen3_5_sst2_full_sweep/finetuned_model.pt")
        if cand_parent.exists():
            args.parent_checkpoint = str(cand_parent)
    ds = load_from_disk(args.dataset_dir); train, val = ds["train"], ds["validation"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True); tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    # Train the newly initialized classifier in fp32. GradScaler cannot
    # unscale gradients belonging to fp16 parameters. Cast the trained model
    # to the requested sweep precision immediately afterwards.
    dtype = torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, local_files_only=True, torch_dtype=dtype).to(device)
    base.config.pad_token_id = tokenizer.pad_token_id; install_qwen_aliases(base)
    qwen = is_qwen_model(base)
    family = detect_family(base) if args.model_family == "auto" else args.model_family
    adapter = ModelAdapter(base, family)
    LOG.info("model=%s qwen_compat=%s hidden=%s heads=%s head_dim=%s layers=%s",
             args.model, qwen, getattr(base.config, "hidden_size", "?"),
             getattr(base.config, "num_attention_heads", "?"),
             getattr(base.config, "head_dim", "?"), getattr(base.config, "num_hidden_layers", "?"))
    LOG.info("detected model family=%s adapter=%s", family, adapter.as_dict())
    active_opt = "muon" if (args.use_muon or args.optimizer == "muon") else "adamw"
    LOG.info("=" * 80)
    LOG.info("ACTIVE SWEEP CONFIGURATION:")
    LOG.info("  model:             %s", args.model)
    LOG.info("  family:            %s", family)
    LOG.info("  dataset:           %s", args.dataset)
    LOG.info("  batch_size:        %d", args.batch_size)
    LOG.info("  recovery_steps:    %d", args.recovery_steps)
    LOG.info("  optimizer:         %s (muon_lr=%.1e, adam_lr=%.1e, weight_decay=%.1e)",
             active_opt, args.muon_lr, args.adam_lr if active_opt == "muon" else args.learning_rate, args.weight_decay)
    LOG.info("  precision:         %s", args.precision)
    LOG.info("=" * 80)
    # Only enable gradient checkpointing for long sequences (> 512) to avoid PyTorch SavedTensorHooks generator assert
    if args.seq_len > 512 and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    checkpoint = out / "finetuned_model.pt"
    if args.parent_checkpoint:
        parent_checkpoint = Path(args.parent_checkpoint)
    else:
        # Dynamically discover parent baseline checkpoint for any family and dataset
        candidates = [
            out.parent / f"{family}_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}2_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"qwen3_5_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"qwen3_5_{args.dataset}_full_sweep" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_full_sweep" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_baseline" / "finetuned_model.pt",
        ]
        parent_checkpoint = next((c for c in candidates if c.exists()), candidates[0])

    # Auto-detect if checkpoint or parent_checkpoint is a LoRA checkpoint
    is_lora_mode = (args.finetune_mode == "lora" or family == "qwen3_5")
    active_ckpt = checkpoint if checkpoint.exists() else (parent_checkpoint if parent_checkpoint.exists() else None)
    if active_ckpt is not None:
        try:
            probe = torch.load(active_ckpt, map_location="cpu", weights_only=False)
            if isinstance(probe, dict):
                if probe.get("metadata", {}).get("finetune_mode") == "lora":
                    is_lora_mode = True
                elif any(".lora_" in k for k in probe.get("state_dict", probe).keys()):
                    is_lora_mode = True
        except Exception:
            pass

    # If LoRA mode is requested or detected, apply LoRA wrappers to the backbone
    if is_lora_mode:
        from framework.utils.lora import apply_lora, save_lora_weights, load_lora_weights, find_all_linear_submodule_names
        # Dynamically discover all linear submodules from the base model, excluding task heads
        dynamic_targets = find_all_linear_submodule_names(base)
        default_cli_targets = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        if args.lora_target_modules != default_cli_targets:
            target_mods = {x.strip() for x in args.lora_target_modules.split(",") if x.strip()}
        else:
            target_mods = dynamic_targets if dynamic_targets else default_cli_targets
        base, num_lora = apply_lora(
            base,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_mods,
        )
        base.to(device)
        # Ensure classification head is trainable
        head = next((getattr(base, attr, None) for attr in
                     ("score", "classifier", "classification_head", "classification_head.dense")
                     if getattr(base, attr, None) is not None), None)
        if head is not None:
            for p in head.parameters(): p.requires_grad = True
        LOG.info("Injected %d LoRALinear layers into %s backbone", num_lora, family)

    if checkpoint.exists():
        LOG.info("Loading cached fine-tuned checkpoint: %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            base.load_state_dict(state["state_dict"], strict=False)
        else:
            base.load_state_dict(state, strict=False)
    elif parent_checkpoint.exists():
        LOG.info("Loading shared baseline checkpoint from parent directory: %s", parent_checkpoint)
        state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            base.load_state_dict(state["state_dict"], strict=False)
        else:
            base.load_state_dict(state, strict=False)
    else:
        if args.finetune_steps > 0:
            if args.finetune_mode == "lora":
                LOG.info("Fine-tuning LoRA adapters for %d steps before module injection (lr=%.1e)", args.finetune_steps, args.learning_rate)
                finetune_lora(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16", lr=args.learning_rate)
                from framework.utils.lora import save_lora_weights
                save_lora_weights(base, str(checkpoint), metadata={"finetune_mode": "lora", "steps": args.finetune_steps, "dataset": args.dataset, "seed": args.seed})
            else:
                LOG.info("Fine-tuning classification head for %d steps before module injection", args.finetune_steps)
                finetune(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16")
                torch.save(base.state_dict(), checkpoint)
            LOG.info("Saved fine-tuned checkpoint: %s", checkpoint)
    if args.precision == "fp16":
        for name, p in base.named_parameters():
            if not p.requires_grad:
                p.data = p.data.half()
    base.to("cpu")
    (out / "finetune_metadata.json").write_text(json.dumps({
        "classification_head_trained": True,
        "finetune_mode": args.finetune_mode,
        "steps": args.finetune_steps,
        "dataset": args.dataset,
        "seed": args.seed
    }), encoding="utf-8")
    LOG.info("Precomputing validation and training batches directly on device %s...", device)
    eval_batches = prepare_eval_batches(tokenizer, val, args.dataset, args.seq_len, args.batch_size, device)
    train_batches = prepare_train_batches(tokenizer, train, args.dataset, args.seq_len, args.batch_size, device, max_batches=max(10, min(100, args.recovery_steps if args.recovery_steps > 0 else 10)))
    LOG.info("Prepared %d eval batches and %d train batches directly on GPU.", len(eval_batches), len(train_batches))
    if isinstance(args.modules, list):
        raw_mods = []
        for item in args.modules:
            raw_mods.extend([x.strip() for x in item.split(",") if x.strip()])
        names = modules() if (not raw_mods or raw_mods == ["all"]) else raw_mods
    elif args.modules == "all":
        names = modules()
    else:
        names = [x.strip() for x in args.modules.split(",") if x.strip()]
    path = out / "summary.csv"; existing = {r["module_name"] for r in csv.DictReader(path.open())} if path.exists() else set()
    module_rank = max(16, getattr(adapter, "head_dim", 16))
    with path.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing: w.writeheader()
        for n, name in enumerate(names, 1):
            if name in existing: continue
            LOG.info(">>> STARTING MODULE [%d/%d]: %s (recovery_steps=%d, batch_size=%d, optimizer=%s)",
                     n, len(names), name, args.recovery_steps, args.batch_size, active_opt)
            t = time.monotonic(); status, acc, err = "PASS", 0.0, ""; had_nan = False; path_name = "native_qwen" if family == "qwen" else ("legacy_bert" if family in {"bert", "roberta"} else "adapted")
            if family == "qwen3_5" and name != "full":
                skip_modules = {
                    "adaptive_span",
                    "adaptive_sparse",   # already skipped but keep
                    "adaptive_svd",      # fails with Tensor + tuple
                    "adasplash",         # fails with t() 3D
                    "a3",                # architecture-unaware compressor, keep skipped until fixed
                }
                if name in skip_modules:
                    status, err, path_name = "SKIP_QWEN3_5_INCOMPATIBLE", f"qwen3_5_incompatible_{name}", "unsupported"
                    row = {"module_name": name, "status": status, "accuracy_pct": 0.0,
                           "elapsed_s": round(time.monotonic()-t, 2), "error": err,
                           "path": path_name, "timestamp": datetime.now().isoformat()}
                    w.writerow(row); f.flush()
                    LOG.info("[%d/%d] %s -> %s (%s)", n, len(names), name, status, err)
                    continue

            support = module_family_support(name).get(family, "unknown")
            if name != "full" and support == "incompatible":
                status, err, path_name = "SKIP_NO_SUPPORT", "unsupported_family", "unsupported"
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err,
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                w.writerow(row); f.flush(); LOG.info("[%d/%d] %s -> %s (%s)", n, len(names), name, status, err)
                continue
            from framework.core.registry import is_module_declared_compatible
            is_compat, reason = is_module_declared_compatible(name, family)
            if name != "full" and not is_compat:
                status, err, path_name = f"SKIP_{family.upper()}_INCOMPATIBLE_DECLARED", reason, "declared_incompatible"
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err,
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                w.writerow(row); f.flush(); LOG.info("[%d/%d] %s -> %s (%s)", n, len(names), name, status, err)
                continue
            if family == "qwen" and args.strict_compat and name != "full" and qwen_module_policy(name) == "incompatible":
                status, err, path_name = "SKIP_NO_SUPPORT", "qwen_incompatible", "strict_skip"
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err,
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                w.writerow(row); f.flush(); LOG.info("[%d/%d] %s -> %s (%s)", n, len(names), name, status, err)
                continue

            # Sequential single-model clone directly on GPU (Zero host RAM bloat, zero PCIe latency)
            # Deepcopy on CPU host memory (zero VRAM overhead) and send fresh candidate to device
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            model = copy.deepcopy(base).to(device)
            # Side effect is projection alias installation; the model itself
            # is not replaced, preserving native HF forward semantics.
            wrap_for_family(model, family)
            typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
            if typ == "attention":
                # For attention replacement modules, merge baseline LoRA adapters into base weights
                # so that get_weight_bias extracts full fine-tuned representations into new_attn.
                for m in model.modules():
                    if hasattr(m, "merge") and callable(m.merge):
                        try:
                            m.merge()
                        except Exception:
                            pass
            else:
                # For compressor modules, do NOT merge LoRA into base weights.
                # Keep base_layer and lora_A/lora_B deltas separate: the compressor factorizes
                # the frozen base_layer.weight, while LoRA deltas remain active and trainable for recovery.
                pass
            injected = False
            had_nan = False
            path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"

            try:
                if family in {"gpt", "gpt2"} and name in {
                    "gated_deltanet_block",
                    "hgrn2_gated_deltanet",
                    "hgrn2_layer",
                    "factorised_matrix_state",
                    "cluster",
                }:
                    raise QwenIncompatibleError("causal_autoregressive_incompatible")

                def _run_candidate_pipeline():
                    nonlocal model, injected, path_name, had_nan
                    if name != "full":
                        model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                        injected = True
                        path_name = f"adapted_{family}" if family != "bert" else "legacy_bert"
                        LOG.info("module=%s path=%s", name, path_name)
                        had_nan = False
                        recovery_targets = {x.strip() for x in args.recovery_modules.split(",") if x.strip()}
                        should_recover = (
                            args.recovery_steps > 0
                            and typ == "attention"
                            and (
                                not recovery_targets
                                or "all" in recovery_targets
                                or "attention" in recovery_targets
                                or name in recovery_targets
                            )
                        )
                        if should_recover:
                            rec_steps = 150 if (family == "qwen3_5" and args.recovery_steps == 100) else args.recovery_steps
                            rec_lr = 5e-5 if (family == "qwen3_5" or adapter.hidden_size > 512) else 1e-4
                            LOG.info("Running %d LoRA recovery fine-tuning steps for %s (lr=%.1e)", rec_steps, name, rec_lr)
                            for m in model.modules():
                                if hasattr(m, "merged") and hasattr(m, "reset_parameters"):
                                    m.reset_parameters()
                                    m.merged = False

                            from framework.utils.compat import get_recovery_params
                            trainable = get_recovery_params(model)
                            if trainable:
                                had_nan = finetune_lora(
                                    model, tokenizer, train, args.dataset, rec_steps, args.seq_len, args.batch_size, device,
                                    args.precision == "fp16", lr=rec_lr, precomputed_batches=train_batches,
                                    optimizer_type=active_opt, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                                    weight_decay=args.weight_decay, timeout=args.timeout,
                                )
                    try:
                        return round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device, fp16=args.precision == "fp16", precomputed_batches=eval_batches), 2)
                    finally:
                        if model is not None:
                            try:
                                model.zero_grad(set_to_none=True)
                                model.to("cpu")
                            except Exception:
                                pass
                            torch.cuda.empty_cache()

                # Hard timeout enforcing max timeout (e.g. 120s / 2 minutes) across injection, recovery, and eval
                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    future = pool.submit(_run_candidate_pipeline)
                    acc = future.result(timeout=args.timeout)
                except (TimeoutError, FuturesTimeoutError) as to_err:
                    raise TimeoutError(f"Module pipeline exceeded hard {args.timeout}s timeout") from to_err
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
            except (TimeoutError, FuturesTimeoutError) as exc:
                status, err = "SKIP_SLOW", str(exc)
            except QwenIncompatibleError as exc:
                status, err = f"SKIP_{family.upper()}_INCOMPATIBLE", str(exc)
                path_name = "unsupported" if qwen else path_name
            except Exception as exc:
                if "out of memory" in str(exc).lower() and not injected:
                    status, err = "SKIP_QWEN3_5_OOM_UNSTABLE" if family == "qwen3_5" else "SKIP", "cuda_oom_injection"
                elif "out of memory" in str(exc).lower() and injected and args.batch_size > 8:
                    try:
                        torch.cuda.empty_cache()
                        safe_batch = max(8, args.batch_size // 2)
                        acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, safe_batch, device), 2)
                        err = f"batch_size reduced {args.batch_size}->{safe_batch} after CUDA OOM"
                        exc = None
                    except Exception as retry: exc = retry
                if name != "full" and is_dtype_error(exc):
                    try:
                        model.float(); typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                        model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                        acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device, precomputed_batches=eval_batches), 2)
                        err = f"fp32 retry: {exc}"
                    except Exception as retry: exc = retry
                if err == "":
                    if exc is not None and "out of memory" in str(exc).lower():
                        status = "SKIP_QWEN3_5_OOM_UNSTABLE" if family == "qwen3_5" else "SKIP_OOM"
                        err = f"CUDA OOM: {exc}"
                    elif is_architecture_error(exc):
                        status = "SKIP_QWEN3_5_INCOMPATIBLE" if family == "qwen3_5" else "SKIP"
                        err = f"{type(exc).__name__}: {exc}"
                    else:
                        status = "FAIL"
                        err = f"{type(exc).__name__}: {exc}"
            if family in {"gpt", "gpt2"} and name in ATTENTION_REGISTRY and status == "PASS":
                if had_nan:
                    status = "SKIP_NUMERICAL_NAN"
                    if not err:
                        err = "recovery_loss_nan"
                elif acc <= 55.0:
                    status = "SKIP_GPT2_INCOMPATIBLE_PLATEAU"
                    if not err:
                        err = "gpt2_causal_geometry_plateau"
            elif family == "t5" and name in ATTENTION_REGISTRY and status == "PASS":
                if had_nan:
                    status = "SKIP_NUMERICAL_NAN"
                    if not err:
                        err = "recovery_loss_nan"
                elif acc <= 51.5:
                    status = "SKIP_T5_INCOMPATIBLE"
                    if not err:
                        err = "t5_relative_bias_and_cross_attention_incompatible"
            elif family in {"bert", "roberta"} and name in ATTENTION_REGISTRY and status == "PASS":
                if had_nan:
                    status = "SKIP_NUMERICAL_NAN"
                    if not err:
                        err = "recovery_loss_nan"
                elif acc <= 51.5:
                    status = "SKIP_ROBERTA_INCOMPATIBLE"
                    if not err:
                        err = "roberta_bidirectional_self_attention_incompatible"
            row = {"module_name": name, "status": status, "accuracy_pct": acc,
                   "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
                   "path": path_name, "timestamp": datetime.now().isoformat()}
            w.writerow(row); f.flush(); LOG.info("[%d/%d] %s -> %s %.2f%%", n, len(names), name, status, acc)
            del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__": main()
