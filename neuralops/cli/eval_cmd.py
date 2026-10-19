# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Evaluation Command

Implements the command-line interface for model evaluation across benchmark datasets.
"""

def cmd_eval(args):
    from evaluation.eval_all_models_datasets import evaluate_model
    print(f"Evaluating {args.model} on {args.dataset}...")
    res = evaluate_model(args.model, dataset=args.dataset)
    print("Evaluation result:", res)
    return 0
