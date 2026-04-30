#!/usr/bin/env python
# studio/cli.py
import argparse
import sys
from pathlib import Path
from studio.config import config, DEVICE
from studio.data.loader import discover_datasets, load_dataset_from_config
from studio.compression.adaptive_compressor import AdvancedZeroShotCompressor
from studio.ensemble.weighted_ensemble import run_smart_ensemble
from studio.routing.smart_router import run_smart_router
from studio.training.trainer import fine_tune_compressed_model
from studio.analysis.evaluate import evaluate_model_on_dataset
from studio.soa.migration import migrate_bert_to_soa, compute_global_basis_from_hidden_states
from studio.utils.logger import setup_logger

logger = setup_logger("SVD-Lin-CLI")

def cmd_compress(args):
    """Compress a model with SVD (zero‑shot or with fine‑tuning)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = args.model
    rank = args.rank
    method = args.method
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=config.model.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config.model.num_labels,
        cache_dir=config.model.cache_dir
    ).to(DEVICE)

    compressor = AdvancedZeroShotCompressor(method=method, base_threshold=config.svd.energy_threshold,
                                             min_rank=config.svd.min_rank)
    model, stats, reduction = compressor.compress_model(model, method)
    logger.info(f"Compressed {stats['layers_compressed']} layers, reduction {reduction:.2f}%")

    if args.finetune:
        # Load dataset (default: from args.dataset or first discovered)
        ds = discover_datasets(config.data.local_glue_path)[0]
        train_ds, val_ds = load_dataset_from_config(ds, tokenizer, config.data.max_seq_length)
        results = fine_tune_compressed_model(model, tokenizer, train_ds, val_ds,
                                             epochs=config.training.epochs,
                                             lr=config.training.learning_rate)
        logger.info(f"Fine‑tuned accuracy: {results['eval_accuracy']:.4f}")

    # Save
    output_dir = Path(args.output) if args.output else Path("./compressed_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def cmd_ensemble(args):
    """Run the smart weighted ensemble on a dataset."""
    run_smart_ensemble(config, args.dataset)

def cmd_router(args):
    """Run the keyword‑based router ensemble."""
    run_smart_router(config, args.dataset)

def cmd_benchmark(args):
    """Benchmark compression methods across ranks and models."""
    from studio.benchmark.universal_svd_benchmark import run_universal_benchmark
    run_universal_benchmark()

def cmd_soa(args):
    """Create and evaluate an SVD‑Optimized Architecture (SOA) model."""
    from transformers import BertModel, AutoTokenizer
    from studio.soa.model import SOABert
    from studio.soa.config import SOAConfig

    bert = BertModel.from_pretrained(args.model).to(DEVICE)
    basis = compute_global_basis_from_hidden_states(bert, k_rank=args.rank)
    cfg = SOAConfig(k_rank=args.rank, num_labels=args.num_labels)
    soa = SOABert(cfg).to(DEVICE)
    soa = migrate_bert_to_soa(bert, soa, basis)
    # Evaluate on a sample dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = discover_datasets(config.data.local_glue_path)[0]
    _, val_ds = load_dataset_from_config(ds, tokenizer, config.data.max_seq_length)
    metrics = evaluate_model_on_dataset(soa, tokenizer, val_ds)
    logger.info(f"SOA accuracy: {metrics['accuracy']:.4f}")

def cmd_generate_data(args):
    """Generate synthetic dataset."""
    from studio.data.synthetic import run_universal_generation
    run_universal_generation(Path(args.output_dir), total_samples_per_domain=args.samples)

def main():
    parser = argparse.ArgumentParser(description="SVD‑Lin Studio – Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compress
    p = subparsers.add_parser("compress", help="Apply SVD compression")
    p.add_argument("--model", default=config.model.name)
    p.add_argument("--rank", type=int, default=config.svd.min_rank)
    p.add_argument("--method", default=config.svd.compression_method)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--output", help="Output directory")

    # ensemble
    p = subparsers.add_parser("ensemble", help="Run weighted ensemble")
    p.add_argument("--dataset", default="./data/test.csv")

    # router
    p = subparsers.add_parser("router", help="Run smart router")
    p.add_argument("--dataset", default="./data")

    # benchmark
    p = subparsers.add_parser("benchmark", help="Run compression benchmark")

    # soa
    p = subparsers.add_parser("soa", help="Build SOA model")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--num_labels", type=int, default=2)

    # generate_data
    p = subparsers.add_parser("generate-data", help="Generate synthetic data")
    p.add_argument("--output_dir", default="./data")
    p.add_argument("--samples", type=int, default=5000)

    args = parser.parse_args()
    if args.command == "compress":
        cmd_compress(args)
    elif args.command == "ensemble":
        cmd_ensemble(args)
    elif args.command == "router":
        cmd_router(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "soa":
        cmd_soa(args)
    elif args.command == "generate-data":
        cmd_generate_data(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()