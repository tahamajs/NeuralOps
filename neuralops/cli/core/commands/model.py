#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Model command family: compress, transplant, calibrate, search, export, generate, serve, train, eval, distill."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

from neuralops.cli.core.errors import ArchitectureMismatchError, ExitCode
from neuralops.cli.core.output import CliReporter


def handle_compress(args: argparse.Namespace) -> int:
    """Executes model compression via the requested engine."""
    from neuralops.cli.compress_cmd import cmd_compress

    try:
        ret = cmd_compress(args)
        return int(ret) if ret is not None else int(ExitCode.SUCCESS)
    except Exception as e_comp:
        err_str = f"{e_comp} {str(getattr(e_comp, '__cause__', '') or '')}"
        if any(
            term in err_str.lower() for term in ["architecture", "could not locate", "unsupported model", "mismatch"]
        ):
            raise ArchitectureMismatchError(
                f"Model architecture incompatible with compressor: {e_comp}. "
                "Ensure the model contains compatible linear layers or attention projections.",
                exit_code=ExitCode.ARCHITECTURE_MISMATCH,
            ) from e_comp
        raise


def handle_transplant(args: argparse.Namespace) -> int:
    """Perform surgical attention transplantation on a model checkpoint."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from neuralops import transplant_attention

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    calib_text = (
        getattr(args, "calibration_text", None)
        or getattr(args, "calibration_dataset", None)
        or getattr(args, "dataset", None)
    )

    if not isinstance(args.rank, int) or isinstance(args.rank, bool) or args.rank < 1:
        raise ValueError("--rank must be a positive integer")
    if (args.calibrate or args.heal) and not calib_text:
        raise ValueError(
            "--calibrate and --heal require --calibration-text / --calibration-dataset so real model inputs are used."
        )
    if args.heal and (
        not isinstance(args.heal_steps, int)
        or isinstance(args.heal_steps, bool)
        or args.heal_steps < 1
        or not math.isfinite(args.heal_lr)
        or args.heal_lr <= 0
    ):
        raise ValueError("--heal-steps and --heal-lr must be positive finite values")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    reporter.log(f"[+] Loading model '{args.model}' for attention transplantation...")
    model = AutoModel.from_pretrained(args.model).to(device)
    teacher = None
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model or args.model)
    calibration_inputs = None
    if args.calibrate or args.heal:
        if args.reference_model:
            teacher = AutoModel.from_pretrained(args.reference_model).to(device).eval()
        else:
            teacher = copy.deepcopy(model).eval()
        sample_text = calib_text
        if sample_text and os.path.isfile(sample_text):
            try:
                with open(sample_text, encoding="utf-8") as f:
                    sample_text = f.read()
            except Exception:
                pass
        sample_texts = [sample_text] if sample_text else ["The quick brown fox jumps over the lazy dog."]
        calibration_inputs = tokenizer(
            sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.seq_len
        )
        calibration_inputs = {k: v.to(device) for k, v in calibration_inputs.items()}

    reporter.log(f"[+] Replacing attention layers with '{args.target}' (rank={args.rank})...")
    res = transplant_attention(
        model,
        target_attention=args.target,
        rank=args.rank,
        calibrate=args.calibrate,
        heal=args.heal,
        teacher=teacher,
        calibration_inputs=calibration_inputs,
        heal_steps=args.heal_steps,
        heal_lr=args.heal_lr,
        device=device,
    )
    reporter.log(f"[✓] Successfully transplanted {res.num_transplanted} attention layers.")
    if res.metrics:
        reporter.log(f"[✓] Surgery metrics: {res.metrics}")

    if args.output:
        reporter.log(f"[+] Saving surgically modified model to '{args.output}'...")
        res.model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        reporter.log(f"[✓] Checkpoint successfully saved to '{args.output}'.")

    return int(ExitCode.SUCCESS)


def handle_search(args: argparse.Namespace) -> int:
    """Executes AutoSVD Fisher & Gram sensitivity search and Pareto rank allocation."""
    import json

    from transformers import AutoModel

    from neuralops.compression_engine.svd.auto_svd import AutoSVDCompressor, solve_optimal_rank_allocation

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    reporter.log(f"Loading model '{args.model}' for AutoSVD Pareto rank search...")
    model = AutoModel.from_pretrained(args.model)

    reporter.log(
        f"Solving optimal rank allocation for target_param_ratio={args.target_param_ratio} (min_rank={args.min_rank})..."
    )
    ranks, stats = solve_optimal_rank_allocation(
        model,
        target_param_ratio=args.target_param_ratio,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )

    if getattr(args, "json", False):
        reporter.print_json({"allocated_ranks": ranks, "stats": stats})
    else:
        reporter.log(f"AutoSVD Pareto Allocation Results for {args.model}:")
        reporter.log(f"  Compression Ratio: {stats.get('compression_ratio', 1.0):.2f}x")
        reporter.log(f"  Param Savings:     {stats.get('param_savings_pct', 0.0):.1f}%")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"allocated_ranks": ranks, "stats": stats}, f, indent=2)
        reporter.log(f"[+] Saved allocated ranks JSON to {args.output_json}")

    if args.save_compressed:
        reporter.log(f"Applying compression and saving model to {args.save_compressed}...")
        compressor = AutoSVDCompressor(
            model,
            target_param_ratio=args.target_param_ratio,
            min_rank=args.min_rank,
            max_rank=args.max_rank,
        )
        c_model, _ = compressor.compress()
        c_model.save_pretrained(args.save_compressed)
        reporter.log(f"[+] Compressed model successfully saved to {args.save_compressed}")

    return int(ExitCode.SUCCESS)


def handle_export(args: argparse.Namespace) -> int:
    """Export model to ONNX, GGUF, SafeTensors, or TorchScript."""
    from transformers import AutoModel

    import neuralops

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    reporter.log(f"Loading base model '{args.model}'...")
    model = AutoModel.from_pretrained(args.model)
    if args.attn:
        reporter.log(f"Applying attention replacement: '{args.attn}' (rank={args.rank})...")
        model = neuralops.replace_attention(model, attention=args.attn, rank=args.rank)
    if args.comp:
        reporter.log(f"Applying model compression: '{args.comp}' (rank={args.rank})...")
        model = neuralops.compress(model, method=args.comp, rank=args.rank)

    from neuralops.export.exporter import export_model

    out = export_model(
        model=model,
        output_dir=args.output_dir,
        format=args.format,
        model_name_or_path=args.model,
        attention_type=args.attn,
        compression_method=args.comp,
        rank=args.rank,
    )
    reporter.log(f"[+] Successfully exported model to {out}")
    return int(ExitCode.SUCCESS)


def handle_generate(args: argparse.Namespace) -> int:
    """Runs autoregressive text generation with self-speculative decoding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder

    reporter = CliReporter(
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        json_mode=getattr(args, "json", False),
    )

    reporter.log(f"Loading generative model '{args.model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    reporter.log(f'Prompt: "{args.prompt}"')

    decoder = SelfSpeculativeDecoder(
        target_model=model,
        max_draft_k=args.draft_k,
        adaptive_k=True,
    )

    t0 = time.time()
    out_ids, stats = decoder.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0
    gen_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    if getattr(args, "json", False):
        reporter.print_json(
            {
                "generated_text": gen_text,
                "elapsed_s": elapsed,
                "tokens_per_sec": len(out_ids[0]) / max(elapsed, 1e-6),
                "stats": stats,
            }
        )
    else:
        print(f"\n{gen_text}\n")
        reporter.log(
            f"Generated {len(out_ids[0])} tokens in {elapsed:.2f}s ({len(out_ids[0])/max(elapsed, 1e-6):.1f} tok/s)"
        )

    return int(ExitCode.SUCCESS)


def handle_serve(args: argparse.Namespace) -> int:
    """Dispatches API inference server or ContinuousBatcher engine."""
    from neuralops.cli.serve_cmd import cmd_serve

    ret = cmd_serve(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_train(args: argparse.Namespace) -> int:
    """Dispatches single or distributed model training."""
    from neuralops.cli.train_cmd import cmd_train

    ret = cmd_train(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_eval(args: argparse.Namespace) -> int:
    """Dispatches model evaluation across accuracy, latency, and perplexity."""
    from neuralops.cli.eval_cmd import cmd_eval

    ret = cmd_eval(args)
    return int(ret) if ret is not None else int(ExitCode.SUCCESS)


def handle_distill(args: argparse.Namespace) -> int:
    """Executes multi-level attention & logit knowledge distillation."""
    from neuralops.cli.distill_cmd import cmd_distill

    try:
        ret = cmd_distill(args)
        return int(ret) if ret is not None else int(ExitCode.SUCCESS)
    except (ImportError, AttributeError):
        # Fallback to direct distillation loop
        import torch
        from transformers import AutoModelForSequenceClassification

        from neuralops.trainer.distillation.multi_level_kd import LogitDistillationLoss

        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher).to(device).eval()
        student = AutoModelForSequenceClassification.from_pretrained(args.student).to(device).train()

        kd_loss_fn = LogitDistillationLoss(mode=args.divergence, temperature=args.temperature)
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

        dummy_input = torch.randint(0, 1000, (args.batch_size, 128), device=device)
        dummy_labels = torch.randint(0, 2, (args.batch_size,), device=device)

        for _ in range(1, 11):
            optimizer.zero_grad()
            with torch.no_grad():
                t_out = teacher(dummy_input)
                t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
            s_out = student(dummy_input)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
            loss = kd_loss_fn(s_logits, t_logits, labels=dummy_labels)
            loss.backward()
            optimizer.step()

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            student.save_pretrained(args.output_dir)
        return int(ExitCode.SUCCESS)


def handle_calibrate(args: argparse.Namespace) -> int:
    """Executes activation covariance calibration or healing."""
    args.calibrate = True
    args.target = getattr(args, "target", getattr(args, "attn", "scaled_dot_product"))
    return handle_transplant(args)
