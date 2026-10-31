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

from neuralops.executor.executor import execute_recipe
from neuralops.executor.planner import plan_recipe
from neuralops.recipe.loader import load_recipe
from neuralops.surgery import ModelSurgery
from neuralops.validator.static import validate_recipe_static


def add_surgery_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Add 'surgery' subcommand parser for declarative recipes."""
    parser = subparsers.add_parser(
        "surgery-recipe",
        aliases=["recipe"],
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
            surgery = ModelSurgery.from_recipe(recipe, model_name_or_path=args.model)
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
