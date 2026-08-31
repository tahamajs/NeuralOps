#!/usr/bin/env python3
"""Cached Qwen sequence-classification quality sweep.

Fine-tunes the Qwen classification head first, then evaluates each canonical
attention/compression module from the same fine-tuned checkpoint.
"""
from __future__ import annotations

import argparse, copy, csv, gc, importlib, json, logging, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "Codes")]
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
    if name == "sst2": return ("sentence", None)
    if name == "mrpc": return ("sentence1", "sentence2")
    return ("sentence1", "sentence2")


def encode(tokenizer, batch, dataset_name, seq_len, device):
    a, b = fields(dataset_name)
    texts = batch[a] if b is None else list(zip(batch[a], batch[b]))
    x = tokenizer(texts, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")
    return {k: v.to(device) for k, v in x.items()}, torch.tensor(batch["label"], device=device)


@torch.inference_mode()
def evaluate(model, tokenizer, ds, name, seq_len, batch_size, device):
    model.eval(); correct = total = 0
    for i in range(0, len(ds), batch_size):
        x, y = encode(tokenizer, {k: ds[k][i:i+batch_size] for k in ds.column_names}, name, seq_len, device)
        logits = model(**x).logits
        correct += int((logits.argmax(-1) == y).sum()); total += len(y)
    return 100.0 * correct / max(1, total)


def finetune(model, tokenizer, train, name, steps, seq_len, batch_size, device, fp16):
    # Head-only fine-tuning is deliberate: it is stable for a randomly added
    # Qwen classifier head and guarantees the head is trained before injection.
    for p in model.parameters(): p.requires_grad = False
    head = getattr(model, "score", None) or getattr(model, "classifier", None)
    if head is None: raise RuntimeError("Qwen sequence-classification head not found")
    for p in head.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    model.train()
    for step in range(steps):
        i = (step * batch_size) % len(train)
        x, y = encode(tokenizer, {k: train[k][i:i+batch_size] for k in train.column_names}, name, seq_len, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            loss = model(**x, labels=y).loss
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if (step + 1) % 25 == 0: LOG.info("fine-tune step %d/%d loss=%.4f", step + 1, steps, float(loss))
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset_dir", required=True); ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--output_dir", required=True); ap.add_argument("--modules", default="all")
    ap.add_argument("--max_steps", type=int, default=1); ap.add_argument("--finetune_steps", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=64); ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16"); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_checkpointing_use_reentrant", default="False")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_compat", action="store_true",
                    help="Skip modules without a truthful native/adapted Qwen path")
    ap.add_argument("--model_family", choices=("auto", "bert", "roberta", "gpt", "qwen", "t5", "ssm", "unknown"), default="auto")
    args = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Qwen quality sweep requires CUDA")
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
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpoint = out / "finetuned_model.pt"
    if checkpoint.exists():
        LOG.info("Loading cached fine-tuned checkpoint: %s", checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        base.load_state_dict(state)
    else:
        LOG.info("Fine-tuning classification head for %d steps before module injection", args.finetune_steps)
        finetune(base, tokenizer, train, args.dataset, args.finetune_steps, args.seq_len, args.batch_size, device, args.precision == "fp16")
        torch.save(base.state_dict(), checkpoint)
        LOG.info("Saved fine-tuned checkpoint: %s", checkpoint)
    if args.precision == "fp16": base.half()
    # Keep exactly one fine-tuned checkpoint in host memory. Module workers
    # receive independent CPU copies, then only the active copy is transferred
    # to CUDA. This avoids reloading 988 MB of weights for every module.
    base.cpu()
    (out / "finetune_metadata.json").write_text(json.dumps({"classification_head_trained": True, "steps": args.finetune_steps, "dataset": args.dataset, "seed": args.seed}), encoding="utf-8")
    names = modules() if args.modules == "all" else [x.strip() for x in args.modules.split(",") if x.strip()]
    path = out / "summary.csv"; existing = {r["module_name"] for r in csv.DictReader(path.open())} if path.exists() else set()
    # Deep-copying CPU tensors is safe to overlap with the active CUDA
    # evaluation. Keep the pool bounded: four full Qwen copies can consume
    # several GB of host RAM, while CUDA work remains serialized.
    try:
        import psutil
        ram_percent = lambda: psutil.virtual_memory().percent
    except ImportError:
        ram_percent = lambda: 0.0
    worker_count = 4 if ram_percent() < 80.0 else 2
    executor = ThreadPoolExecutor(max_workers=worker_count)
    module_rank = max(16, adapter.head_dim) if family == "qwen" else 16
    pending = {}
    def queue_copy(module_name):
        if module_name not in pending:
            pending[module_name] = executor.submit(copy.deepcopy, base)
    for candidate in names:
        if candidate not in existing:
            queue_copy(candidate)
            if len(pending) >= 4: break
    with path.open("a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing: w.writeheader()
        for n, name in enumerate(names, 1):
            if name in existing: continue
            if worker_count == 4 and ram_percent() >= 80.0:
                LOG.warning("Host RAM is %.1f%%; reducing CPU prefetch workers 4 -> 2", ram_percent())
                for future in pending.values(): future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                pending.clear(); worker_count = 2
                executor = ThreadPoolExecutor(max_workers=worker_count)
            t = time.monotonic(); status, acc, err = "PASS", 0.0, ""; path_name = "native_qwen" if family == "qwen" else ("legacy_bert" if family in {"bert", "roberta"} else "adapted")
            support = module_family_support(name).get(family, "unknown")
            if name != "full" and support == "incompatible":
                status, err, path_name = "SKIP_NO_SUPPORT", "unsupported_family", "unsupported"
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
            queue_copy(name)
            model = pending.pop(name).result().to(device)
            # Side effect is projection alias installation; the model itself
            # is not replaced, preserving native HF forward semantics.
            wrap_for_family(model, family)
            if n < len(names) and names[n] not in existing:
                queue_copy(names[n])
            injected = False
            try:
                if name != "full":
                    typ = "attention" if name in ATTENTION_REGISTRY else "compressor"
                    model, _ = apply_module_to_model(model, name, typ, rank=module_rank, device=device)
                    injected = True
                    path_name = "adapted_qwen" if qwen else "legacy_bert"
                    LOG.info("module=%s path=%s", name, path_name)
                acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device), 2)
            except QwenIncompatibleError as exc:
                status, err = "SKIP_QWEN_INCOMPATIBLE", str(exc)
                path_name = "unsupported" if qwen else path_name
            except Exception as exc:
                if "out of memory" in str(exc).lower() and not injected:
                    status, err = "SKIP", "cuda_oom_injection"
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
                        acc = round(evaluate(model, tokenizer, val, args.dataset, args.seq_len, args.batch_size, device), 2)
                        err = f"fp32 retry: {exc}"
                    except Exception as retry: exc = retry
                if err == "":
                    status = "SKIP" if is_architecture_error(exc) else "FAIL"; err = f"{type(exc).__name__}: {exc}"
            row = {"module_name": name, "status": status, "accuracy_pct": acc,
                   "elapsed_s": round(time.monotonic()-t, 2), "error": err[:1000],
                   "path": path_name, "timestamp": datetime.now().isoformat()}
            w.writerow(row); f.flush(); LOG.info("[%d/%d] %s -> %s %.2f%%", n, len(names), name, status, acc)
            del model; gc.collect(); torch.cuda.empty_cache()
    executor.shutdown(wait=True)

if __name__ == "__main__": main()
