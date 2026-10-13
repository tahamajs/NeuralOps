"""`compress` command."""
def cmd_compress(args):
    from neuralops.core.compression_pipeline import compress_model
    print(f"Compressing model {args.model} via {args.method} (rank={args.rank})...")
    res = compress_model(args.model, method=args.method, rank=args.rank)
    print("Compression complete:", res)
    return 0
