"""Unified neuralops entry point.

Usage:
    python -m neuralops <command> [options]

Commands:
    info        Show project information & GPU status
    list        List registered canonical attention & compressor modules
    bench       Run pipeline or KD benchmark on any registered module
    train       Train a model
    compress    Compress a model
    eval        Evaluate a model
    test        Run test suite
    serve       Start API server
    doctor      Diagnose installation
"""
import argparse
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        prog="neuralops",
        description="NeuralOps unified CLI & benchmark suite",
    )
    parser.add_argument("--version", action="version", version="neuralops 1.2.0")

    sub = parser.add_subparsers(dest="command", required=True)

    # info
    sub.add_parser("info", help="Show project information & GPU status")

    # list
    p_list = sub.add_parser("list", help="List modules")
    p_list.add_argument(
        "--type", choices=["attention", "compressor", "all"], default="all"
    )
    p_list.add_argument("--json", action="store_true", help="Output as JSON")

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmarks on any module")
    p_bench.add_argument("--module", "--target", dest="module", required=False, default=None, help="Module key from registry")
    p_bench.add_argument("--all-attention", action="store_true", help="Benchmark all 130 attention mechanisms")
    p_bench.add_argument("--all-compressors", action="store_true", help="Benchmark all 72 compression engines")
    p_bench.add_argument("--all", action="store_true", help="Benchmark all attention and compressor modules (202 total)")
    p_bench.add_argument("--mode", choices=["eval", "kd"], default="eval", help="Benchmark mode: eval (pipeline) or kd (distillation)")
    p_bench.add_argument("--model", default="gpt2", help="Base model architecture (default: gpt2)")
    p_bench.add_argument("--dataset", default="sst2", help="Dataset name (default: sst2)")
    p_bench.add_argument("--rank", type=int, default=32, help="Rank for compression/LoRA (default: 32)")
    p_bench.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    p_bench.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    p_bench.add_argument("--recovery_steps", type=int, default=50, help="KD recovery steps")
    p_bench.add_argument("--self_train_steps", type=int, default=20, help="Stage 2 self train steps")
    p_bench.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    p_bench.add_argument("--device", default="cuda", help="Target device (cuda or cpu)")
    p_bench.add_argument("--output-dir", "--output_dir", default=None, help="Directory to save sweep benchmark results")
    p_bench.add_argument("--parallel", type=int, default=1, help="Parallel execution workers")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--config", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="Compress a model")
    p_compress.add_argument("--model", required=True)
    p_compress.add_argument("--method", required=True)
    p_compress.add_argument("--rank", type=int, default=32)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--dataset", default="sst2")

    # test
    sub.add_parser("test", help="Run test suite")

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--port", type=int, default=8000)

    # doctor
    sub.add_parser("doctor", help="Diagnose installation")

    return parser


def dispatch(args):
    if args.command == "info":
        from neuralops.cli.info import cmd_info
        return cmd_info()
    elif args.command == "list":
        from neuralops.cli.list_cmd import cmd_list
        return cmd_list(args)
    elif args.command == "bench":
        from neuralops.cli.bench_cmd import cmd_bench
        return cmd_bench(args)
    elif args.command == "train":
        from neuralops.cli.train_cmd import cmd_train
        return cmd_train(args)
    elif args.command == "compress":
        from neuralops.cli.compress_cmd import cmd_compress
        return cmd_compress(args)
    elif args.command == "eval":
        from neuralops.cli.eval_cmd import cmd_eval
        return cmd_eval(args)
    elif args.command == "test":
        from neuralops.cli.test_cmd import cmd_test
        return cmd_test()
    elif args.command == "serve":
        from neuralops.cli.serve_cmd import cmd_serve
        return cmd_serve(args)
    elif args.command == "doctor":
        from neuralops.cli.doctor import cmd_doctor
        return cmd_doctor()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args) or 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
