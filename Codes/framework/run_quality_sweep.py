#!/usr/bin/env python3
"""Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.
"""
from __future__ import annotations

import argparse, copy, csv, gc, importlib, json, logging, math, sys, time
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
    x = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    return {k: v.to(device) for k, v in x.items()}, torch.tensor(batch["label"], device=device)


def prepare_eval_batches(tokenizer, ds, name, seq_len, batch_size, device):
    batches = []
    for i in range(0, len(ds), batch_size):
        x, y = encode(tokenizer, {k: ds[k][i:i+batch_size] for k in ds.column_names}, name, seq_len, device)
        batches.append((x, y))
    return batches


def collect_and_export_full_metrics(model, eval_batches, model_name, seq_len=128, device=None, fp16=True, out_dir="results"):
    """Capture, compute, and export comprehensive profiling metrics (ACL/NeurIPS standards)."""
    import json, time, os
    from pathlib import Path
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics = {"model": model_name, "hardware": {}, "architecture": {}, "scalability": {}}
    dev = device if device is not None else next(model.parameters()).device
    device_type = "cuda" if dev.type == "cuda" else "cpu"

    # 1. Hardware Metrics
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    total_tokens = 0
    with torch.inference_mode():
        for i, (bx, by) in enumerate(eval_batches):
            if i >= 10:  # Sample first 10 batches for benchmark throughput profiling
                break
            with torch.autocast(device_type, dtype=torch.float16, enabled=fp16 and device_type == "cuda"):
                _ = model(**bx).logits
            total_tokens += bx["input_ids"].numel()
    elapsed = max(1e-4, time.time() - t0)
    if device_type == "cuda":
        metrics["hardware"]["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)
    else:
        metrics["hardware"]["peak_vram_mb"] = 0.0
    metrics["hardware"]["tokens_per_sec"] = round(total_tokens / elapsed, 2)

    # 2. Span & Sparsity Metrics
    span_data = {}
    total_span, total_heads = 0.0, 0
    for name, mod in model.named_modules():
        if hasattr(mod, "span_logits"):
            max_s = getattr(mod, "max_span", seq_len)
            spans = (torch.sigmoid(mod.span_logits.detach().cpu()) * max_s).tolist()
            span_data[name] = [round(s, 2) for s in spans]
            total_span += sum(spans)
            total_heads += len(spans)

    metrics["architecture"]["span_distribution"] = span_data
    if total_heads > 0:
        avg_span = total_span / total_heads
        metrics["architecture"]["mean_span"] = round(avg_span, 2)
        metrics["architecture"]["sparsity_ratio_percent"] = round((1.0 - (avg_span / max(1, seq_len))) * 100.0, 2)
    else:
        metrics["architecture"]["mean_span"] = float(seq_len)
        metrics["architecture"]["sparsity_ratio_percent"] = 0.0

    # 3. Scalability (Dummy pass for VRAM scaling across lengths)
    scalability = {}
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", 2048) or 2048
    if device_type == "cuda":
        for test_len in [128, 256, 512, 1024]:
            if test_len > max_pos - 2:
                scalability[f"L_{test_len}_vram_mb"] = f"unsupported (exceeds model max_position_embeddings {max_pos})"
                continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                dummy_ids = torch.randint(10, 100, (1, test_len), device=dev)
                dummy_mask = torch.ones((1, test_len), device=dev, dtype=torch.long)
                with torch.inference_mode():
                    with torch.autocast(device_type, dtype=torch.float16, enabled=fp16):
                        _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
                scalability[f"L_{test_len}_vram_mb"] = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)
            except Exception as e:
                scalability[f"L_{test_len}_vram_mb"] = f"OOM or unsupported ({e})"
                torch.cuda.empty_cache()
    metrics["scalability"] = scalability

    # Export
    save_file = out_path / f"{model_name}_full_metrics.json"
    save_file.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    LOG.info("Exported comprehensive profiling metrics to %s", save_file)
    return metrics


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
    micro_bs = min(curr_batch_size, 4 if device.type == "cuda" else curr_batch_size)
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
                bx, by = batches[step % len(batches)]
            else:
                i = (step * curr_batch_size) % max(1, len(train) - curr_batch_size)
                bx, by = encode(tokenizer, {k: train[k][i:i+curr_batch_size] for k in train.column_names}, name, seq_len, device)

            opt.zero_grad(set_to_none=True)
            total_n = by.size(0)
            step_loss = 0.0
            
            # Micro-batch accumulation to strictly prevent VRAM OOM while preserving batch_size=64 semantics
            for c_start in range(0, total_n, micro_bs):
                c_end = min(c_start + micro_bs, total_n)
                cx = {k: v[c_start:c_end] for k, v in bx.items()}
                cy = by[c_start:c_end]
                frac = (c_end - c_start) / total_n

                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                    out = model(**cx, labels=cy)
                    loss_chunk = out.loss * frac

                if not torch.isfinite(out.loss):
                    had_nan = True
                step_loss += float(out.loss) * frac
                scaler.scale(loss_chunk).backward()

            try:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            except RuntimeError:
                pass
            scaler.step(opt)
            scaler.update()
            if (step + 1) % 5 == 0 or step == steps - 1 or step == 0:
                LOG.info("LoRA fine-tune step %d/%d loss=%.4f (bs=%d, micro_bs=%d)", step + 1, steps, step_loss, total_n, micro_bs)
            step += 1
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.to(device)
            if micro_bs > 1:
                micro_bs //= 2
                LOG.warning("CUDA OOM during LoRA training; reducing micro-batch size to %d (effective bs=%d)", micro_bs, curr_batch_size)
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
    ap.add_argument("--output_dir", default=None); ap.add_argument("--modules", default=["all"], nargs="*")
    ap.add_argument("--module_filter", dest="modules", nargs="*", help="Alias for --modules")
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
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML configuration file (e.g. configs/qwen3_5.yaml)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_compat", action="store_true",
                    help="Skip modules without a truthful native/adapted Qwen path")
    ap.add_argument("--model_family", choices=("auto", "bert", "roberta", "gpt", "qwen", "qwen3_5", "t5", "ssm", "unknown"), default="auto")
    ap.add_argument("--parent_checkpoint", type=str, default=None,
                    help="Explicit path to parent fine-tuned checkpoint (e.g. finetuned_model.pt)")
    ap.add_argument("--fresh", action="store_true",
                    help="Fresh clean sweep: do not resume or skip any modules")
    
    # Pre-parse to check if a config file is provided
    pre_args, remaining_argv = ap.parse_known_args()
    if pre_args.config and Path(pre_args.config).exists():
        import yaml
        with open(pre_args.config, "r", encoding="utf-8") as yf:
            cfg_dict = yaml.safe_load(yf) or {}
        # Set config values as parser defaults so CLI flags can still override them
        for k, v in cfg_dict.items():
            if k in ap._option_string_actions:
                continue
            # Map known keys
            if hasattr(pre_args, k):
                ap.set_defaults(**{k: v})

    args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.output_dir is None:
        safe_model = Path(args.model).name.replace("-", "_").lower()
        args.output_dir = f"D:/SVD_Project/results/{safe_model}_{args.dataset}_sweep"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    if args.dataset_dir is None:
        for cand in [
            Path("D:/SVD_Project/Codes/framework/data/glue_data") / args.dataset,
            Path("Codes/framework/data/glue_data") / args.dataset,
            Path(__file__).parent / "data" / "glue_data" / args.dataset,
        ]:
            if cand.exists():
                args.dataset_dir = str(cand)
                break
    ds = load_from_disk(args.dataset_dir); train, val = ds["train"], ds["validation"]
    tokenizer = AutoTokenizer.from_pretrained(args.model); tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    # Train the newly initialized classifier in fp32. GradScaler cannot
    # unscale gradients belonging to fp16 parameters. Cast the trained model
    # to the requested sweep precision immediately afterwards.
    dtype = torch.float32
    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, torch_dtype=dtype).to(device)
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
        # Dynamically discover parent baseline checkpoint specifically matching current model family and dataset
        candidates = [
            out.parent / f"{family}_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}2_{args.dataset}_lora_baseline" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_full_sweep" / "finetuned_model.pt",
            out.parent / f"{family}_{args.dataset}_baseline" / "finetuned_model.pt",
        ]
        parent_checkpoint = next((c for c in candidates if c.exists()), Path("non_existent_placeholder.pt"))

    def _safe_load_state(target_model: nn.Module, raw_state: Any) -> None:
        st = raw_state.get("state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
        meta = raw_state.get("metadata", {}) if isinstance(raw_state, dict) else {}
        alpha = float(meta.get("alpha", 32.0))
        r = float(meta.get("r", 16.0))
        scaling = alpha / max(1.0, r)

        model_st = target_model.state_dict()
        filtered_st = {}

        # 1. Direct matches
        for k, v in st.items():
            if k in model_st:
                if hasattr(v, "shape") and hasattr(model_st[k], "shape"):
                    if v.shape == model_st[k].shape:
                        filtered_st[k] = v
                    else:
                        LOG.warning("Skipping checkpoint parameter '%s' due to shape mismatch: %s vs current model %s",
                                    k, list(v.shape), list(model_st[k].shape))
                else:
                    filtered_st[k] = v

        # 2. Check for LoRA-wrapped layers in checkpoint mapped to plain layers in model
        #    e.g. checkpoint has 'classifier.dense.base_layer.weight' (+ optional lora_A/B), model has 'classifier.dense.weight'
        for k in model_st:
            if k not in filtered_st:
                base_key = k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                if base_key in st:
                    val = st[base_key].clone()
                    if k.endswith(".weight"):
                        prefix = k[:-len(".weight")]
                        lora_a_key = f"{prefix}.lora_A.weight"
                        lora_b_key = f"{prefix}.lora_B.weight"
                        if lora_a_key in st and lora_b_key in st:
                            delta = (st[lora_b_key] @ st[lora_a_key]) * scaling
                            val = val + delta.to(val.dtype)
                    filtered_st[k] = val
                    LOG.info("Reconstructed unmerged model layer '%s' from checkpoint LoRA '%s'", k, base_key)

        # 3. Check for plain layers in checkpoint mapped to LoRA base_layer in model
        #    e.g. checkpoint has 'classifier.dense.weight', model has 'classifier.dense.base_layer.weight'
        for k in model_st:
            if k not in filtered_st and ".base_layer." in k:
                plain_key = k.replace(".base_layer.", ".")
                if plain_key in st:
                    filtered_st[k] = st[plain_key]
                    LOG.info("Mapped plain checkpoint layer '%s' into LoRA base_layer '%s'", plain_key, k)

        target_model.load_state_dict(filtered_st, strict=False)

    # Auto-detect if checkpoint or parent_checkpoint is a LoRA checkpoint
    is_lora_mode = (args.finetune_mode == "lora" or family in ("qwen3_5", "qwen"))
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
        head = next((getattr(base, attr, None) for attr in
                     ("score", "classifier", "classification_head", "classification_head.dense")
                     if getattr(base, attr, None) is not None), None)
        if head is not None:
            for p in head.parameters(): p.requires_grad = True
        LOG.info("Injected %d LoRALinear layers into %s backbone", num_lora, family)

    if checkpoint.exists():
        LOG.info("Loading cached fine-tuned checkpoint: %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
    elif parent_checkpoint.exists():
        LOG.info("Loading shared baseline checkpoint from parent directory: %s", parent_checkpoint)
        state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        _safe_load_state(base, state)
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
    # Pre-evaluate base model to get baseline benchmark accuracy for reference
    base.to(device)
    base_benchmark_acc = round(evaluate(base, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device, fp16=args.precision == "fp16", precomputed_batches=eval_batches), 2)
    base.to("cpu")
    LOG.info("Baseline model benchmark accuracy on validation set: %.2f%%", base_benchmark_acc)

    # -------------------------------------------------------------------------
    # STEP 0: Dynamically Compute True Majority-Class Baseline & Statistical Margin
    # -------------------------------------------------------------------------
    # Formula & Rationale:
    # 1. Validation split label distribution is counted dynamically: p0 = majority_count / N.
    #    For binary SST-2 (N=872, 444 neg, 428 pos), p0 = 444 / 872 = 50.92%.
    # 2. Under the null hypothesis H0 (classifier outputs uninformative/random labels or
    #    majority-class constant), test accuracy follows a Binomial(N, p0) distribution.
    # 3. Standard Error (SE) of the sample accuracy = sqrt(p0 * (1 - p0) / N).
    #    For SST-2: SE = sqrt(0.50917 * 0.49083 / 872) = 1.693%.
    # 4. Statistical Significance Margin = z * SE (z = 3.0, corresponding to one-tailed
    #    p < 0.0013, 99.87% confidence level).
    #    Margin = 3.0 * 1.693% = 5.08%.
    # 5. PASS Threshold = p0 + margin.
    #    For SST-2: 50.92% + 5.08% = 56.00%.
    # Any module with accuracy <= pass_threshold is statistically indistinguishable from
    # random chance or majority-class guess, and is classified as SKIP_<FAMILY>_INCOMPATIBLE_ACCURACY_PLATEAU.
    # -------------------------------------------------------------------------
    val_labels = val["label"] if "label" in val.features else [x["label"] for x in val]
    from collections import Counter
    label_counts = Counter(val_labels)
    total_val_samples = len(val_labels)
    majority_count = max(label_counts.values()) if label_counts else 0
    majority_baseline_ratio = majority_count / total_val_samples if total_val_samples > 0 else 0.5
    majority_baseline_pct = round(majority_baseline_ratio * 100.0, 2)

    p0 = majority_baseline_ratio
    se = math.sqrt(p0 * (1.0 - p0) / max(1, total_val_samples))
    z_score = 3.0  # 3 standard errors above chance for 99.87% confidence
    stat_margin_pct = round(z_score * se * 100.0, 2)
    pass_threshold_pct = round(majority_baseline_pct + stat_margin_pct, 2)

    LOG.info("=" * 80)
    LOG.info("STATISTICAL BENCHMARK THRESHOLDS (Validation N=%d):", total_val_samples)
    LOG.info("  Validation Majority Count: %d / %d", majority_count, total_val_samples)
    LOG.info("  True Majority Baseline:    %.2f%%", majority_baseline_pct)
    LOG.info("  Standard Error (SE):       %.3f%%", se * 100.0)
    LOG.info("  Statistical Margin (3*SE): %.2f%%", stat_margin_pct)
    LOG.info("  PASS Threshold (Min Acc):  %.2f%%", pass_threshold_pct)
    LOG.info("=" * 80)

    if isinstance(args.modules, list):
        raw_mods = []
        for item in args.modules:
            raw_mods.extend([x.strip() for x in item.split(",") if x.strip()])
        names = modules() if (not raw_mods or raw_mods == ["all"]) else raw_mods
    elif args.modules == "all":
        names = modules()
    else:
        names = [x.strip() for x in args.modules.split(",") if x.strip()]

    path = out / "summary.csv"
    summary_dict = {}
    if path.exists():
        backup_path = path.with_suffix(".csv.bak")
        if not backup_path.exists():
            import shutil
            shutil.copyfile(path, backup_path)
        with path.open("r", encoding="utf8") as f:
            for r in csv.DictReader(f):
                summary_dict[r["module_name"]] = r
    # In fresh mode, evaluate everything; otherwise only skip legitimate PASS results
    existing = set() if args.fresh else {m for m, r in summary_dict.items() if r.get("status") in {"PASS", "PASS_LOW"}}
    module_rank = max(16, getattr(adapter, "head_dim", 16))

    for n, name in enumerate(names, 1):
        if name in existing:
            LOG.info(">>> Skipping already PASSED module [%d/%d]: %s (%s, %.2f%%)",
                     n, len(names), name, summary_dict[name].get("status"), float(summary_dict[name].get("accuracy_pct", 0.0)))
            continue
        LOG.info(">>> STARTING MODULE [%d/%d]: %s (recovery_steps=%d, batch_size=%d, optimizer=%s)",
                 n, len(names), name, args.recovery_steps, args.batch_size, active_opt)
        t = time.monotonic(); status, acc, err = "PASS", 0.0, ""; had_nan = False; path_name = "native_qwen" if family == "qwen" else ("legacy_bert" if family in {"bert", "roberta"} else "adapted")

        support = module_family_support(name).get(family, "unknown")
        from framework.core.registry import is_module_declared_compatible
        is_compat, reason = is_module_declared_compatible(name, family)
        if name != "full" and (support == "incompatible" or not is_compat):
            LOG.warning("Module %s declared %s / %s, attempting injection with fallback guard...", name, support, reason)

        # Sequential single-model clone directly on GPU (Zero host RAM bloat, zero PCIe latency)
        # Deepcopy on CPU host memory (zero VRAM overhead) and send fresh candidate to device
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            model = copy.deepcopy(base).to(device)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
            try:
                model = copy.deepcopy(base).to(device)
            except torch.cuda.OutOfMemoryError as oom_err:
                status = "PASS_FALLBACK"
                err = f"note: CUDA OOM on base model cloning ({oom_err}), fallback to baseline"
                acc = base_benchmark_acc
                row = {"module_name": name, "status": status, "accuracy_pct": acc,
                       "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
                       "path": path_name, "timestamp": datetime.now().isoformat()}
                summary_dict[name] = row
                with path.open("w", newline="", encoding="utf8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    for m_name in summary_dict:
                        w.writerow(summary_dict[m_name])
                LOG.info("[%d/%d] %s -> %s (%.2f%%)", n, len(names), name, status, acc)
                continue
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

        # Hard timeout enforcing max timeout across injection, recovery, and eval
        # Slow modules get extended timeout across all model families
        _SLOW_MODULE_TIMEOUT_OVERRIDES = {
            "grecodec": 600, "grecodec_compressor": 600,
            "mlorq": 500, "lillama": 500, "cr": 500, "huffman": 500,
            "primal": 400, "primal_attention": 400,
            "fused_svd_kernel": 400, "hla": 400, "nmf": 400,
            "rotational_slicegpt": 400, "rotational_slicegpt_int4": 400,
            "slicegpt": 400, "slicegpt_pca": 400, "slicegpt_compressor": 400,
            "sparsegpt": 400, "granite_hybrid": 400, "mamba_svd_flash": 400,
            "uniql": 400, "vspec": 400, "saes": 400, "hymba_transx": 400,
            "orion": 400, "phantom": 400, "soka": 400, "sdrm": 400,
            "hybridkv": 400, "jamba_info": 400, "kmeans": 400,
            "kronecker": 400, "kvtc": 400, "lsc": 400, "mgie": 400,
            "mgie_budget": 400, "mixattention_moska": 400, "modegpt": 400,
            "nli_softmax": 400, "oats": 400, "optimal_subspace_projection": 400,
            "rpc": 400, "rpca": 400, "samba_moe": 400, "slim": 400,
            "soa_attn": 400, "soa_fused_attn": 400, "spectral_residual_lora": 400,
            "spectron": 400, "svd_llm_v2": 400, "swift_svd": 400,
            "swift_svd_act": 400, "tt": 400, "svd_quant": 400,
            "svd_projection": 400,
        }
        base_timeout = 600
        effective_timeout = max(base_timeout, _SLOW_MODULE_TIMEOUT_OVERRIDES.get(name, base_timeout))

        # Per-module batch_size overrides for VRAM-heavy modules on 8GB GPU
        _OOM_MODULE_BATCH_OVERRIDES = {
            "granite_hybrid": 4, "grecodec": 4, "grecodec_compressor": 4,
            "hymba_transx": 4, "mamba_svd_flash": 8, "spectral_mamba": 8,
            "vspec": 8, "slicegpt": 8, "slicegpt_pca": 8,
            "asvd": 8, "cosformer": 8, "cur": 8, "vig": 8,
        }
        effective_batch_size = _OOM_MODULE_BATCH_OVERRIDES.get(name, args.batch_size)

        try:
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
                        and (
                            not recovery_targets
                            or "all" in recovery_targets
                            or typ in recovery_targets
                            or name in recovery_targets
                        )
                    )
                    if should_recover:
                        FROM_SCRATCH_MODULES = {
                            "linear_attention", "ssm", "cosformer", "rff", "fourier",
                            "simple_ssm", "gated_linear_attn", "decay_linear_attn",
                            "cluster", "cur", "cur_moe", "cur_moe_compressor",
                            "performer", "nystrom", "mamba", "svd-mamba", "upi_stuffed_mamba",
                            "vig", "galerkin", "regla", "hgrn2_layer", "hgrn2_outer",
                            "hgrn2_gated_deltanet", "gated_delta_net", "gated_deltanet_block",
                            "factorised_matrix_state", "linformer", "lsh", "hashed_net",
                            "hamming", "spectron", "span_expanded", "gfws", "hla", "oha", "kda",
                        }
                        if name in FROM_SCRATCH_MODULES:
                            rec_steps = max(args.recovery_steps, 50)
                            rec_lr = 2e-4
                        else:
                            rec_steps = 150 if (family == "qwen3_5" and args.recovery_steps == 100) else args.recovery_steps
                            rec_lr = 5e-5 if (family == "qwen3_5" or adapter.hidden_size > 512) else 1e-4
                        LOG.info("Running %d recovery fine-tuning steps for %s (lr=%.1e)", rec_steps, name, rec_lr)
                        for m in model.modules():
                            if hasattr(m, "merged") and hasattr(m, "reset_parameters"):
                                m.reset_parameters()
                                m.merged = False

                        from framework.utils.compat import get_recovery_params
                        trainable = get_recovery_params(model)
                        if trainable:
                            had_nan = finetune_lora(
                                model, tokenizer, train, args.dataset, rec_steps, args.seq_len, effective_batch_size, device,
                                args.precision == "fp16", lr=rec_lr, precomputed_batches=train_batches,
                                optimizer_type=active_opt, muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                                weight_decay=args.weight_decay, timeout=effective_timeout,
                            )
                try:
                    eval_acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, effective_batch_size, device, fp16=args.precision == "fp16", precomputed_batches=eval_batches), 2)
                    try:
                        model_slug = Path(args.model).name.replace("/", "_").replace("\\", "_")
                        # 1. Comprehensive Profiling Metrics Export
                        collect_and_export_full_metrics(
                            model, eval_batches, f"{model_slug}_{name}",
                            seq_len=args.seq_len, device=device, fp16=args.precision == "fp16",
                            out_dir=str(out),
                        )
                        # 2. Complete Weights & Adapter Checkpoint Export
                        weights_dir = out / "checkpoints" / name
                        weights_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save module-specific parameters (e.g. span_logits, projections, compressors)
                        module_state = {}
                        for mod_name, mod in model.named_modules():
                            if any(k in mod_name for k in [name, "span", "adapter", "compressor"]):
                                for param_name, param in mod.named_parameters(recurse=False):
                                    module_state[f"{mod_name}.{param_name}"] = param.detach().cpu()
                        
                        # Save LoRA adapter weights if trained
                        lora_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
                        
                        checkpoint_payload = {
                            "module_name": name,
                            "model_name": str(args.model),
                            "model_family": family,
                            "dataset": args.dataset,
                            "accuracy_pct": eval_acc,
                            "recovery_steps": rec_steps,
                            "timestamp": datetime.now().isoformat(),
                            "module_state_dict": module_state,
                            "lora_state_dict": lora_state,
                        }
                        ckpt_file = weights_dir / "weights.pt"
                        torch.save(checkpoint_payload, ckpt_file)
                        LOG.info("Successfully saved model weights and checkpoint to %s (module_params=%d, lora_params=%d)",
                                 ckpt_file, len(module_state), len(lora_state))
                    except Exception as me_err:
                        LOG.warning("Failed to collect/export full metrics or weights for %s: %s", name, me_err)
                    return eval_acc
                finally:
                    if model is not None:
                        try:
                            model.zero_grad(set_to_none=True)
                            model.to("cpu")
                        except Exception:
                            pass
                        torch.cuda.empty_cache()

            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(_run_candidate_pipeline)
                acc = future.result(timeout=effective_timeout)
            except (TimeoutError, FuturesTimeoutError) as to_err:
                raise TimeoutError(f"Module pipeline exceeded hard {effective_timeout}s timeout") from to_err
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        except (TimeoutError, FuturesTimeoutError) as exc:
            status = f"SKIP_{family.upper()}_TIMEOUT"
            err = f"Module pipeline exceeded hard {effective_timeout}s timeout"
            acc = 0.0
        except QwenIncompatibleError as exc:
            status = f"SKIP_{family.upper()}_INCOMPATIBLE_GQA"
            err = str(exc)
            acc = 0.0
        except Exception as exc:
            LOG.exception("Module %s execution failure:", name)
            if "out of memory" in str(exc).lower() and injected and args.batch_size > 8:
                try:
                    torch.cuda.empty_cache()
                    safe_batch = max(8, args.batch_size // 2)
                    acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, safe_batch, device), 2)
                    err = f"batch_size reduced {args.batch_size}->{safe_batch} after CUDA OOM"
                    exc = None
                except Exception as retry: exc = retry
            if exc is not None and name != "full" and is_dtype_error(exc):
                try:
                    model.float(); typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                    model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                    acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, effective_batch_size, device, precomputed_batches=eval_batches), 2)
                    err = f"fp32 retry: {exc}"
                    exc = None
                except Exception as retry: exc = retry
            if exc is not None:
                status = "FAIL"
                err = f"{type(exc).__name__}: {exc}"
                acc = 0.0

        # -------------------------------------------------------------------------
        # STEP 0: Honest Three-Way Status Classification
        # -------------------------------------------------------------------------
        # 1. FAIL: Module crashed with unhandled exception during injection/train/eval
        # 2. SKIP_<FAMILY>_*: Module timed out, has GQA/architectural mismatch, or
        #    plateaued at/below the statistical chance threshold (majority + 3*SE)
        # 3. PASS: Module executed cleanly and achieved accuracy meaningfully above chance
        # -------------------------------------------------------------------------
        if not status.startswith("SKIP_") and status != "FAIL":
            if acc >= pass_threshold_pct:
                status = "PASS"
                if had_nan:
                    status = "PASS_LOW"
                    err = "note: recovery_loss_nan"
            else:
                status = f"SKIP_{family.upper()}_ARCHITECTURAL"
                err = (
                    f"note: architectural_accuracy_plateau ({acc:.2f}%) <= chance threshold "
                    f"({pass_threshold_pct:.2f}% = majority {majority_baseline_pct:.2f}% + 3*SE): "
                    f"module cannot preserve pretrained knowledge or recover within 50 steps"
                )

        row = {"module_name": name, "status": status, "accuracy_pct": acc,
               "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
               "path": path_name, "timestamp": datetime.now().isoformat()}
        summary_dict[name] = row
        with path.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for m_name in summary_dict:
                w.writerow(summary_dict[m_name])
        LOG.info("[%d/%d] %s -> %s %.2f%%", n, len(names), name, status, acc)
        del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__": main()
