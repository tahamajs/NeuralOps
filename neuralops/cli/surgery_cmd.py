#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Command for NeuralOps Recipe Language and Model Surgery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional

def execute_recipe(*args: Any, **kwargs: Any) -> Any:
    from neuralops.executor.executor import execute_recipe as _impl
    return _impl(*args, **kwargs)


def plan_recipe(*args: Any, **kwargs: Any) -> Any:
    from neuralops.executor.planner import plan_recipe as _impl
    return _impl(*args, **kwargs)


def load_recipe(*args: Any, **kwargs: Any) -> Any:
    from neuralops.recipe.loader import load_recipe as _impl
    return _impl(*args, **kwargs)


def validate_recipe_static(*args: Any, **kwargs: Any) -> Any:
    from neuralops.validation.static import validate_recipe_static as _impl
    return _impl(*args, **kwargs)


def _get_model_surgery():
    from neuralops.surgery import ModelSurgery
    return ModelSurgery


def add_surgery_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add 'surgery' subcommand parser for declarative recipes."""
    parser = subparsers.add_parser(
        "surgery-recipe",
        help="Plan, validate, and execute declarative NeuralOps transformation recipes",
    )
    cmd_subparsers = parser.add_subparsers(dest="surgery_action", help="Action to execute")

    # plan
    plan_p = cmd_subparsers.add_parser("plan", help="Compile and inspect execution plan for a recipe")
    plan_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")
    plan_p.add_argument("--json", action="store_true", help="Output plan as JSON")

    # validate
    val_p = cmd_subparsers.add_parser("validate", help="Statically validate recipe rules W1-W15")
    val_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")

    # apply
    apply_p = cmd_subparsers.add_parser("apply", help="Execute recipe transformations against target model")
    apply_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")
    apply_p.add_argument("--model", type=str, default=None, help="Override target model name or path")
    apply_p.add_argument("--output", type=str, default=None, help="Destination directory to save transformed model")
    apply_p.add_argument("--profile", type=str, default=None, help="Override validation profile")
    apply_p.add_argument("--json", action="store_true", help="Output execution evidence as JSON")


def build_surgery_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Build unified surgery argument parser for declarative recipes and model surgery."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops surgery",
        description="Plan, validate, and execute declarative NeuralOps transformation recipes",
    )
    cmd_subparsers = parser.add_subparsers(dest="surgery_action", help="Action to execute")

    # plan
    plan_p = cmd_subparsers.add_parser("plan", help="Compile and inspect execution plan for a recipe")
    plan_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")
    plan_p.add_argument("--json", action="store_true", help="Output plan as JSON")

    # validate
    val_p = cmd_subparsers.add_parser("validate", help="Statically validate recipe rules W1-W15")
    val_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")

    # apply
    apply_p = cmd_subparsers.add_parser("apply", help="Execute recipe transformations against target model")
    apply_p.add_argument("recipe", type=str, help="Path to recipe YAML/JSON file")
    apply_p.add_argument("--model", type=str, default=None, help="Override target model name or path")
    apply_p.add_argument("--output", type=str, default=None, help="Destination directory to save transformed model")
    apply_p.add_argument("--profile", type=str, default=None, help="Override validation profile")
    apply_p.add_argument("--json", action="store_true", help="Output execution evidence as JSON")

    return parser


def build_transplant_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Build argument parser for attention transplantation and micro-healing."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops transplant",
        description="Replace attention, optionally calibrate/heal, and save a surgical checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--attention",
        "--target",
        "--target-attention",
        dest="attention",
        type=str,
        required=True,
        help="Target attention mechanism (e.g. svd_linear_attention, gla_attention, cosformer)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Low-rank bottleneck dimension (default: 64)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Run layerwise Ridge alignment calibration (requires --calibration-text / --dataset)",
    )
    parser.add_argument(
        "--heal",
        action="store_true",
        default=False,
        help="Run fast layerwise gradient micro-healing after replacement",
    )
    parser.add_argument(
        "--heal-steps",
        type=int,
        default=25,
        help="Number of micro-healing iterations per layer (default: 25)",
    )
    parser.add_argument(
        "--heal-lr",
        type=float,
        default=2e-3,
        help="Learning rate for layerwise micro-healing (default: 2e-3)",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Optional teacher checkpoint; defaults to a copy of --model",
    )
    parser.add_argument(
        "--calibration-text",
        "--calibration-dataset",
        "--dataset",
        dest="calibration_text",
        type=str,
        default=None,
        help="Text or dataset identifier used to create real tokenizer inputs for calibration/healing",
    )
    parser.add_argument(
        "--output-dir",
        "--save-dir",
        "--output",
        dest="output_dir",
        type=str,
        default="models/transplanted",
        help="Directory for the surgical model and manifest (default: models/transplanted)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output execution evidence as JSON",
    )
    return parser


def cmd_transplant(args: argparse.Namespace) -> int:
    """Handler for attention transplantation subcommand."""
    import copy
    import math
    import os
    import torch
    from transformers import AutoModel, AutoTokenizer
    from neuralops import transplant_attention

    calib_text = getattr(args, "calibration_text", None) or getattr(args, "calibration_dataset", None) or getattr(args, "dataset", None)

    if not isinstance(args.rank, int) or isinstance(args.rank, bool) or args.rank < 1:
        raise ValueError("--rank must be a positive integer")
    if (args.calibrate or args.heal) and not calib_text:
        raise ValueError("--calibrate and --heal require --calibration-text / --calibration-dataset so real model inputs are used.")
    if args.heal and (
        not isinstance(args.heal_steps, int)
        or isinstance(args.heal_steps, bool)
        or args.heal_steps < 1
        or not math.isfinite(args.heal_lr)
        or args.heal_lr <= 0
    ):
        raise ValueError("--heal-steps and --heal-lr must be positive finite values")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Loading model '{args.model}' for attention transplantation...")
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
                with open(sample_text, "r", encoding="utf-8") as f:
                    sample_text = f.read(4096)
            except Exception:
                pass
        encoded = tokenizer(sample_text or "The quick brown fox jumps over the lazy dog.", return_tensors="pt")
        calibration_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }

    print(f"[+] Transplanting attention '{args.attention}' (rank={args.rank}, calibrate={args.calibrate})...")
    transplanted = transplant_attention(model, args.attention, rank=args.rank)

    if args.calibrate:
        from neuralops.core._attention_surgery import calibrate_attention_subspace
        transplanted = calibrate_attention_subspace(teacher, transplanted, calibration_inputs)

    if getattr(args, "heal", False):
        print(f"[+] Running layerwise subspace micro-healing (steps={args.heal_steps}, lr={args.heal_lr})...")
        from neuralops.core._attention_surgery import heal_replaced_model
        heal_stats = heal_replaced_model(
            teacher_model=teacher,
            student_model=transplanted,
            calibration_inputs=calibration_inputs,
            steps=args.heal_steps,
            lr=args.heal_lr,
            also_ridge_init=True,
        )
        print(f"[✓] Micro-healing completed: Initial MSE={heal_stats['initial_mse']:.4f} -> Final MSE={heal_stats['final_mse']:.4f}")

    from neuralops.models.surgical_io import save_surgical_checkpoint
    out_dir = getattr(args, "output_dir", None) or getattr(args, "output", "models/transplanted")
    print(f"[+] Saving surgical model checkpoint to '{out_dir}'...")
    save_surgical_checkpoint(transplanted, out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[✓] Transplantation complete. Surgical checkpoint saved to '{out_dir}'.")
    return 0


def cmd_surgery(args: argparse.Namespace) -> int:
    """Unified handler for surgery subcommand."""
    action = getattr(args, "surgery_action", None)
    if action:
        return handle_surgery_command(args)
    if getattr(args, "model", None) and getattr(args, "attention", None):
        return cmd_transplant(args)
    print("Please specify surgery action: plan, validate, or apply. See --help.", file=sys.stderr)
    return 0


def handle_surgery_command(args: argparse.Namespace) -> int:
    action = getattr(args, "surgery_action", None)
    if not action:
        print("Please specify action: plan, validate, or apply. See --help.", file=sys.stderr)
        return 1

    recipe_path = args.recipe
    recipe = load_recipe(recipe_path)

    if action == "validate":
        try:
            warnings = validate_recipe_static(recipe)
            print(f"✓ Recipe '{recipe.metadata.name}' passed static validation (W1-W15).")
            if warnings:
                print(f"  Warnings ({len(warnings)}):")
                for w in warnings:
                    print(f"   - {w}")
            return 0
        except Exception as e:
            print(f"✗ Static validation failed: {e}", file=sys.stderr)
            return 1

    elif action == "plan":
        try:
            plan = plan_recipe(recipe)
            if getattr(args, "json", False):
                print(json.dumps({
                    "recipe_name": plan.recipe_name,
                    "target_model": plan.target_model,
                    "stages_count": len(plan.stages),
                    "stages": [
                        {
                            "id": s.id,
                            "operation": s.operation_type,
                            "version": s.version,
                            "config": s.config,
                            "depends_on": s.depends_on,
                        }
                        for s in plan.stages
                    ],
                }, indent=2))
            else:
                print(f"Execution Plan: {plan.recipe_name} (target: {plan.target_model})")
                print(f"Total Stages: {len(plan.stages)}")
                for idx, s in enumerate(plan.stages, 1):
                    deps = f" [depends on {s.depends_on}]" if s.depends_on else ""
                    print(f"  {idx}. Stage '{s.id}': {s.operation_type}@{s.version}{deps}")
            return 0
        except Exception as e:
            print(f"✗ Planning failed: {e}", file=sys.stderr)
            return 1

    elif action == "apply":
        try:
            print(f"Executing recipe '{recipe.metadata.name}'...")
            surgery = _get_model_surgery().from_recipe(recipe, model_name_or_path=args.model)
            if args.profile:
                surgery.recipe.validation.profile = args.profile

            surgery.apply()
            evidence = surgery.get_evidence()

            if getattr(args, "json", False):
                print(json.dumps(evidence, indent=2))
            else:
                print(f"✓ Transformation completed successfully. Total evidence records: {len(evidence)}")
                for ev in evidence:
                    op = ev.get("operation", "unknown")
                    ratio = ev.get("compression_ratio")
                    if ratio:
                        print(f"  - {op}: compression ratio = {ratio:.4f}")

            if args.output:
                surgery.save_pretrained(args.output)
                print(f"✓ Model saved to {args.output}")

            return 0
        except Exception as e:
            print(f"✗ Recipe execution failed: {e}", file=sys.stderr)
            return 1

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="nops-surgery", description="NeuralOps Model Surgery CLI")
    subparsers = parser.add_subparsers(dest="command")
    add_surgery_subparser(subparsers)
    args = parser.parse_args(argv)
    return handle_surgery_command(args)


if __name__ == "__main__":
    sys.exit(main())
