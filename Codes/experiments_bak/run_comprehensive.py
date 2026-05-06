# #!/usr/bin/env python3
# import argparse
# import yaml
# import pandas as pd
# import torch
# import gc
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer
# from pathlib import Path
# from datasets import load_dataset

# # Import only the parts of your framework that exist
# from framework.models.compressed_bert import CompressedBertModel
# from framework.trainer.trainer_svd import train_model, benchmark_model, count_parameters
# from framework.analysis.plotting import create_comprehensive_plots
# from framework.utils.checkpoint import CheckpointManager
# from framework.utils.device import get_device
# from framework.config.experiment_config import ExperimentConfig

# # ----------------------------------------------------------------------
# # Local GLUE Dataset (replaces missing framework.data.dataset_utils)
# # ----------------------------------------------------------------------
# class GLUEDataset(Dataset):
#     def __init__(self, df, tokenizer, max_length, is_regression=False):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.is_regression = is_regression

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         text = str(self.df.iloc[idx]['text'])
#         label = self.df.iloc[idx]['label']
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         item = {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'labels': torch.tensor(label, dtype=torch.float if self.is_regression else torch.long)
#         }
#         return item

# def load_glue_dataset(task_name, local_cache_dir, max_samples=None, seed=42):
#     """Load a GLUE task and return train/val DataFrames with 'text' and 'label'."""
#     dataset = load_dataset("glue", task_name, cache_dir=local_cache_dir)

#     def format_text(example):
#         if task_name in ["mrpc", "qnli", "rte", "wnli"]:
#             return example["sentence1"] + " [SEP] " + example["sentence2"]
#         elif task_name in ["mnli", "mnli_matched", "mnli_mismatched"]:
#             return example["premise"] + " [SEP] " + example["hypothesis"]
#         else:  # cola, sst2, stsb
#             return example["sentence"]

#     train_df = dataset["train"].to_pandas()
#     val_df = dataset["validation"].to_pandas()

#     train_df["text"] = train_df.apply(format_text, axis=1)
#     val_df["text"] = val_df.apply(format_text, axis=1)

#     if task_name == "stsb":
#         train_df["label"] = train_df["label"].astype(float)
#         val_df["label"] = val_df["label"].astype(float)
#         is_regression = True
#     else:
#         train_df["label"] = train_df["label"].astype(int)
#         val_df["label"] = val_df["label"].astype(int)
#         is_regression = False

#     if max_samples and len(train_df) > max_samples:
#         train_df = train_df.sample(n=max_samples, random_state=seed)
#     if max_samples and len(val_df) > max_samples:
#         val_df = val_df.sample(n=max_samples, random_state=seed)

#     return train_df, val_df, is_regression

# def setup_logger():
#     import logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     return logging.getLogger(__name__)

# def run_experiment(config, device, logger):
#     # Create output directories
#     output_dir = Path(config.output_dir)
#     plots_dir = output_dir / "plots"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     plots_dir.mkdir(parents=True, exist_ok=True)

#     checkpoint_manager = CheckpointManager(output_dir)

#     # Load GLUE tasks from config
#     with open(config.config_file, 'r') as f:
#         yaml_config = yaml.safe_load(f)
#     glue_tasks = yaml_config.get('glue_tasks', ["rte", "cola", "sst2", "mrpc", "qnli", "wnli", "stsb"])
#     if config.selected_datasets:
#         glue_tasks = [t for t in glue_tasks if t in config.selected_datasets]

#     if not glue_tasks:
#         logger.error("❌ No GLUE tasks selected!")
#         return pd.DataFrame()

#     experiment_id, all_results = checkpoint_manager.load_checkpoint()
#     logger.info(f"🔄 Starting from experiment ID: {experiment_id}")

#     models_to_test = list(config.base_models.items())
#     if config.selected_models:
#         models_to_test = [(k, v) for k, v in models_to_test if k in config.selected_models]
#     ranks_to_test = config.svd_ranks
#     if config.selected_ranks:
#         ranks_to_test = [r for r in ranks_to_test if r in config.selected_ranks]

#     local_glue_path = yaml_config.get('local_glue_path', './datasets/glue')

#     for model_name, model_path in models_to_test:
#         logger.info(f"\n{'='*60}\n🧠 TESTING MODEL: {model_name.upper()} ({model_path})\n{'='*60}")
#         for task_name in glue_tasks:
#             logger.info(f"\n   📝 DATASET: GLUE/{task_name}")
#             try:
#                 train_df, val_df, is_regression = load_glue_dataset(
#                     task_name,
#                     local_cache_dir=local_glue_path,
#                     max_samples=config.max_samples,
#                     seed=config.seed
#                 )

#                 num_labels = 1 if is_regression else len(train_df["label"].unique())
#                 tokenizer = AutoTokenizer.from_pretrained(model_path)

#                 train_dataset = GLUEDataset(train_df, tokenizer, config.max_length, is_regression)
#                 val_dataset = GLUEDataset(val_df, tokenizer, config.max_length, is_regression)

#                 train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#                 val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

#                 for rank in ranks_to_test:
#                     experiment_id += 1
#                     logger.info(f"\n      🔬 Rank {rank}:")
#                     try:
#                         model = CompressedBertModel(model_path, rank=rank, num_labels=num_labels).to(device)
#                         params_after = count_parameters(model)
#                         logger.info(f"         ✅ Model Initialized | Params: {params_after:,}")
#                         compressed_stats = benchmark_model(model, val_loader, device)
#                         compressed_results = train_model(
#                             model, train_loader, val_loader, device, config,
#                             num_labels, is_regression=is_regression
#                         )
#                         result = {
#                             'experiment_id': experiment_id,
#                             'model': model_name,
#                             'dataset': task_name,
#                             'task_type': 'regression' if is_regression else 'classification',
#                             'num_labels': num_labels,
#                             'rank': rank,
#                             'compressed_accuracy': compressed_results['final_accuracy'],
#                             'compressed_params': params_after,
#                             'compressed_batch_time': compressed_stats['avg_batch_time'],
#                             'compressed_memory_mb': compressed_stats['memory_usage_mb'],
#                         }
#                         all_results.append(result)
#                         metric = 'Corr' if is_regression else 'Acc'
#                         logger.info(f"         ✅ Complete | {metric}: {compressed_results['final_accuracy']:.4f}")

#                         if experiment_id % config.save_checkpoint_interval == 0:
#                             checkpoint_manager.save_checkpoint(experiment_id, all_results)

#                         del model
#                         gc.collect()
#                         if device.type == 'mps':
#                             torch.mps.empty_cache()
#                         elif device.type == 'cuda':
#                             torch.cuda.empty_cache()
#                     except Exception as e:
#                         logger.error(f"         ❌ Error in rank {rank}: {e}")
#                         continue
#                 del tokenizer
#                 gc.collect()
#             except Exception as e:
#                 logger.error(f"   ❌ Error processing GLUE/{task_name}: {e}")
#                 continue

#     results_df = pd.DataFrame(all_results)
#     results_path = output_dir / "compression_results.csv"
#     results_df.to_csv(results_path, index=False)
#     logger.info(f"\n✅ All experiments completed! Results saved to {results_path}")

#     if not results_df.empty:
#         create_comprehensive_plots(results_df, plots_dir)
#     return results_df

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='SVD Compression on GLUE Tasks')
#     parser.add_argument('--config', type=str, default='./framework/config/default_config.yaml', help='Path to YAML config file')
#     parser.add_argument('--models', type=str, nargs='+', choices=['bert', 'distilbert', 'roberta'], help='Models to test')
#     parser.add_argument('--datasets', type=str, nargs='+', help='GLUE task names')
#     parser.add_argument('--ranks', type=int, nargs='+', help='SVD ranks to test')
#     parser.add_argument('--max-samples', type=int, help='Max samples per dataset')
#     parser.add_argument('--batch-size', type=int, help='Batch size')
#     parser.add_argument('--epochs', type=int, help='Training epochs')
#     parser.add_argument('--seed', type=int, help='Random seed')
#     parser.add_argument('--output-dir', type=str, help='Output directory')
#     return parser.parse_args()

# def main():
#     args = parse_arguments()

#     # Load YAML config
#     with open(args.config, 'r') as f:
#         yaml_cfg = yaml.safe_load(f)

#     # Create an ExperimentConfig object (assumes default constructor)
#     config = ExperimentConfig()
#     # Populate from YAML + command line overrides
#     config.config_file = args.config
#     config.max_length = yaml_cfg.get('max_seq_length', 128)
#     config.batch_size = args.batch_size or yaml_cfg.get('batch_size', 16)
#     config.epochs = args.epochs or yaml_cfg.get('epochs', 3)
#     config.learning_rate = yaml_cfg.get('learning_rate', 2e-5)
#     config.seed = args.seed or yaml_cfg.get('seed', 42)
#     config.max_samples = args.max_samples or yaml_cfg.get('max_train_samples', 500)
#     config.output_dir = args.output_dir or yaml_cfg.get('output_dir', './results')
#     config.save_checkpoint_interval = 5
#     config.val_split = 0.2

#     # Models and ranks
#     config.base_models = {
#         'bert': 'bert-base-uncased',
#         'distilbert': 'distilbert-base-uncased',
#         'roberta': 'roberta-base'
#     }
#     config.svd_ranks = [16, 32, 64, 128]  # default
#     if args.ranks:
#         config.svd_ranks = args.ranks
#     config.selected_models = args.models or []
#     config.selected_datasets = args.datasets or []
#     config.selected_ranks = []

#     # Setup device and logger (replaces setup_experiment)
#     device = get_device()
#     logger = setup_logger()
#     logger.info(f"⚡ Using device: {device}")

#     # Run
#     try:
#         results_df = run_experiment(config, device, logger)
#         if not results_df.empty:
#             logger.info("\n" + "="*80)
#             logger.info("📊 EXPERIMENT SUMMARY")
#             logger.info("="*80)
#             logger.info(f"Average Accuracy: {results_df['compressed_accuracy'].mean():.4f}")
#             best = results_df.loc[results_df['compressed_accuracy'].idxmax()]
#             logger.info(f"\n🏆 Best Overall:")
#             logger.info(f"   Model: {best['model']}, Dataset: {best['dataset']}, Rank: {best['rank']}, Acc: {best['compressed_accuracy']:.4f}")
#     except KeyboardInterrupt:
#         logger.info("\n⚠️ Interrupted")
#     except Exception as e:
#         logger.error(f"\n❌ Failed: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()