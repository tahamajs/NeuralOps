"""`eval` command."""
def cmd_eval(args):
    from neuralops.evaluation.eval_all_models_datasets import evaluate_model
    print(f"Evaluating {args.model} on {args.dataset}...")
    res = evaluate_model(args.model, dataset=args.dataset)
    print("Evaluation result:", res)
    return 0
