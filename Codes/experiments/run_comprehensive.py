#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path

from framework.config.experiment_config import ExperimentConfig, setup_experiment
from framework.data.dataset_discovery import discover_datasets, MultiChoiceDataset, SingleLabelDataset
from framework.models.compressed_bert import CompressedBertModel
from framework.training.trainer_svd import train_model, benchmark_model, count_parameters
from framework.analysis.plotting import create_comprehensive_plots
from framework.utils.checkpoint import CheckpointManager

def run_experiment(config: ExperimentConfig, device, logger):
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.plots_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(config.output_dir)
    
    logger.info("🔍 Discovering datasets...")
    config.datasets = discover_datasets(config)
    if config.selected_datasets:
        config.datasets = [d for d in config.datasets if d.name in config.selected_datasets]
    if not config.datasets:
        logger.error("❌ No valid datasets found!")
        return pd.DataFrame()
    
    logger.info(f"🎯 Selected datasets: {[d.name for d in config.datasets]}")
    experiment_id, all_results = checkpoint_manager.load_checkpoint()
    logger.info(f"🔄 Starting from experiment ID: {experiment_id}")
    
    models_to_test = list(config.base_models.items())
    if config.selected_models:
        models_to_test = [(k, v) for k, v in models_to_test if k in config.selected_models]
    ranks_to_test = config.svd_ranks
    if config.selected_ranks:
        ranks_to_test = [r for r in ranks_to_test if r in config.selected_ranks]
    
    for model_name, model_path in models_to_test:
        logger.info(f"\n{'='*60}\n🧠 TESTING MODEL: {model_name.upper()} ({model_path})\n{'='*60}")
        for dataset_config in config.datasets:
            logger.info(f"\n   📝 DATASET: {dataset_config.name} (task_type={dataset_config.task_type}, num_labels={dataset_config.num_labels})")
            try:
                df = pd.read_csv(dataset_config.path)
                if len(df) > config.max_samples:
                    df = df.sample(n=config.max_samples, random_state=config.seed)
                split_idx = int(len(df) * (1 - config.val_split))
                train_df = df.iloc[:split_idx].reset_index(drop=True)
                val_df = df.iloc[split_idx:].reset_index(drop=True)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                is_multichoice = (dataset_config.task_type == 'mc')
                
                if is_multichoice:
                    train_dataset = MultiChoiceDataset(train_df, tokenizer, config.max_length)
                    val_dataset = MultiChoiceDataset(val_df, tokenizer, config.max_length)
                    if len(train_dataset) == 0 or len(val_dataset) == 0:
                        logger.error(f"❌ No valid samples for {dataset_config.name} after filtering.")
                        continue
                else:
                    train_dataset = SingleLabelDataset(train_df, tokenizer, config.max_length)
                    val_dataset = SingleLabelDataset(val_df, tokenizer, config.max_length)
                
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
                
                for rank in ranks_to_test:
                    experiment_id += 1
                    logger.info(f"\n      🔬 Rank {rank}:")
                    try:
                        model = CompressedBertModel(model_path, rank=rank, num_labels=dataset_config.num_labels).to(device)
                        params_after = count_parameters(model)
                        logger.info(f"         ✅ Model Initialized | Params: {params_after:,}")
                        compressed_stats = benchmark_model(model, val_loader, device)
                        compressed_results = train_model(model, train_loader, val_loader, device, config, dataset_config.num_labels, is_multichoice=is_multichoice)
                        result = {
                            'experiment_id': experiment_id,
                            'model': model_name,
                            'dataset': dataset_config.name,
                            'task_type': dataset_config.task_type,
                            'num_labels': dataset_config.num_labels,
                            'rank': rank,
                            'compressed_accuracy': compressed_results['final_accuracy'],
                            'compressed_params': params_after,
                            'compressed_batch_time': compressed_stats['avg_batch_time'],
                            'compressed_memory_mb': compressed_stats['memory_usage_mb'],
                        }
                        all_results.append(result)
                        logger.info(f"         ✅ Complete | Acc: {compressed_results['final_accuracy']:.4f}")
                        if experiment_id % config.save_checkpoint_interval == 0:
                            checkpoint_manager.save_checkpoint(experiment_id, all_results)
                        del model
                        gc.collect()
                        if device.type == 'mps': torch.mps.empty_cache()
                        elif device.type == 'cuda': torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"         ❌ Error in rank {rank}: {e}")
                        continue
                del tokenizer
                gc.collect()
            except Exception as e:
                logger.error(f"   ❌ Error processing dataset {dataset_config.name}: {e}")
                continue
    
    results_df = pd.DataFrame(all_results)
    results_path = Path(config.output_dir) / "compression_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✅ All experiments completed! Results saved to {results_path}")
    if not results_df.empty:
        create_comprehensive_plots(results_df, config.plots_dir)
    return results_df

def parse_arguments():
    parser = argparse.ArgumentParser(description='SVD Compression Experiments for Reasoning Tasks')
    parser.add_argument('--models', type=str, nargs='+', choices=['bert', 'distilbert', 'roberta'], help='Models to test')
    parser.add_argument('--datasets', type=str, nargs='+', help='Datasets to test')
    parser.add_argument('--ranks', type=int, nargs='+', help='SVD ranks to test')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples per dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = ExperimentConfig(
        selected_models=args.models if args.models else [],
        selected_datasets=args.datasets if args.datasets else [],
        selected_ranks=args.ranks if args.ranks else [],
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir
    )
    device, logger = setup_experiment(config)
    try:
        results_df = run_experiment(config, device, logger)
        if not results_df.empty:
            logger.info("\n" + "="*80)
            logger.info("📊 EXPERIMENT SUMMARY")
            logger.info("="*80)
            logger.info(f"Average Accuracy: {results_df['compressed_accuracy'].mean():.4f}")
            best_overall = results_df.loc[results_df['compressed_accuracy'].idxmax()]
            logger.info(f"\n🏆 Best Overall Accuracy:")
            logger.info(f"   Model: {best_overall['model']}, Dataset: {best_overall['dataset']}, Rank: {best_overall['rank']}, Acc: {best_overall['compressed_accuracy']:.4f}")
    except KeyboardInterrupt:
        logger.info("\n⚠️ Experiment interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Experiment failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()