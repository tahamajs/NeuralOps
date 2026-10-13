"""`bench` command."""
def cmd_bench(args):
    print(f"Benchmarking {args.module} on {args.model}/{args.dataset}...")
    from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
    res = evaluate_single_combination_opt(
        model_name=args.model,
        attention_module=args.module,
        dataset_name=args.dataset,
        rank=args.rank,
    )
    print("Benchmark complete:", res)
    return 0
