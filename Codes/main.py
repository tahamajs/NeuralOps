#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗   ██╗██████╗     ██╗     ██╗███╗   ██╗                         ║
║   ██╔════╝██║   ██║██╔══██╗    ██║     ██║████╗  ██║                         ║
║   ███████╗██║   ██║██║  ██║    ██║     ██║██╔██╗ ██║                         ║
║   ╚════██║╚██╗ ██╔╝██║  ██║    ██║     ██║██║╚██╗██║                         ║
║   ███████║ ╚████╔╝ ██████╔╝    ███████╗██║██║ ╚████║                         ║
║   ╚══════╝  ╚═══╝  ╚═════╝     ╚══════╝╚═╝╚═╝  ╚═══╝                         ║
║                                                                               ║
║   SVD-Lin: Linearizing Transformer Attention via Truncated SVD                ║
║   =========================================================================   ║
║                                                                               ║
║   A complete framework for SVD-based attention compression in transformers   ║
║                                                                               ║
║   Features:                                                                   ║
║   • Multiple attention mechanisms (SVD, Nyström, Performer, Linformer)        ║
║   • Compression methods (standard_svd, randomized_svd, quantized_svd, etc.)   ║
║   • GLUE benchmark integration with local dataset support                     ║
║   • Hyperparameter optimization with Optuna                                   ║
║   • Interactive visualizations and HTML reports                               ║
║   • Multi-seed experiments with result aggregation                            ║
║   • MPS/CUDA/CPU support with memory optimizations                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

# Add plugin discovery
from framework import discover_plugins, list_all_components


def add_plugin_arguments(parser):
    """Add plugin-related arguments to parser."""
    parser.add_argument("--list_components", action="store_true",
                        help="List all available components and exit")
    parser.add_argument("--plugin_dir", type=str, default=None,
                        help="Additional plugin directory to scan")
    parser.add_argument("--extend", action="store_true",
                        help="Enable extension mode (load custom plugins)")


# ============================================================================
# Framework Imports
# ============================================================================

# Core utilities
from framework.utils import (
    get_device,
    set_seed,
    setup_logger,
    clear_memory,
    count_parameters,
    format_number,
    format_time,
    CheckpointManager,
    ModelProfiler,
    measure_inference_latency,
    detect_architecture,
    get_layer_patterns,
    save_json,
    load_json,
)

# Data loading
from framework.data import (
    load_glue_task,
    load_glue_tasks,
    get_glue_task_info,
    get_num_labels,
    get_metric_name,
    get_text_columns,
    preprocess_function,
    tokenize_dataset,
    DataCollator,
    MultiChoiceDataset,
    is_multichoice_dataset,
)

# Compression
from framework.compression import (
    SVDLinearLayer,
    AdvancedZeroShotCompressor,
    AdaptiveTransformerCompressor,
    get_adaptive_ranks,
    apply_adaptive_svd,
)

# Attentions
from framework.attentions import (
    SVDLinearAttention,
    NystromAttention,
    PerformerAttention,
    LinformerAttention,
    RALALinearAttention,
    HybridAttention,
)

# Training
from framework.training import (
    SVDAwareTrainer,
    create_trainer,
    HyperparameterOptimizer,
    train_model,
    evaluate_model,
    benchmark_model,
    compute_metrics,
)

# Visualization
from framework.visualization import (
    SVDAttentionVisualizer,
    AdvancedReportGenerator,
    generate_csv_summary,
    generate_html_report,
    plot_accuracy_vs_rank,
    plot_training_curves,
)

# Routing
from framework.routing import (
    TaskRouter,
    RouterEnsemble,
    KeywordRouter,
    detect_task_type,
)

# SOA (SVD-Optimized Architecture)
from framework.soa import (
    SOAConfig,
    SOABert,
    compute_global_basis_from_bert,
    migrate_bert_to_soa,
    compress_bert_true_svd,
)

# ============================================================================
# Configuration
# ============================================================================

class MainConfig:
    """Main configuration class for the framework."""
    
    def __init__(self, args):
        self.args = args
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or use defaults."""
        if self.args.config and os.path.exists(self.args.config):
            with open(self.args.config, 'r') as f:
                if self.args.config.endswith('.yaml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            self._update_from_dict(config)
        else:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Hardware
        self.device = get_device(self.args.device if hasattr(self.args, 'device') else "auto")
        self.force_cpu_for_svd = True
        self.gradient_checkpointing = True
        self.mixed_precision = False
        
        # Data
        self.local_glue_path = self.args.local_glue_path if hasattr(self.args, 'local_glue_path') else "/Users/tahamajs/Documents/uni/Project/datasets/glue"
        self.max_seq_length = self.args.max_seq_length if hasattr(self.args, 'max_seq_length') else 128
        self.batch_size = self.args.batch_size if hasattr(self.args, 'batch_size') else 16
        self.eval_batch_size = self.args.eval_batch_size if hasattr(self.args, 'eval_batch_size') else 32
        
        # Model
        self.model_name = self.args.model_name if hasattr(self.args, 'model_name') else "bert-base-uncased"
        self.attention_type = self.args.attention_type if hasattr(self.args, 'attention_type') else "svd"
        self.rank = self.args.rank if hasattr(self.args, 'rank') else 128
        
        # Training
        self.epochs = self.args.epochs if hasattr(self.args, 'epochs') else 3
        self.learning_rate = self.args.learning_rate if hasattr(self.args, 'learning_rate') else 2e-5
        self.weight_decay = self.args.weight_decay if hasattr(self.args, 'weight_decay') else 0.01
        self.warmup_ratio = self.args.warmup_ratio if hasattr(self.args, 'warmup_ratio') else 0.1
        self.max_grad_norm = self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 1.0
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps if hasattr(self.args, 'gradient_accumulation_steps') else 1
        
        # Experiment
        self.tasks = self.args.tasks if hasattr(self.args, 'tasks') else ["cola", "sst2", "mrpc"]
        self.seed = self.args.seed if hasattr(self.args, 'seed') else 42
        self.num_seeds = self.args.num_seeds if hasattr(self.args, 'num_seeds') else 1
        self.output_dir = self.args.output_dir if hasattr(self.args, 'output_dir') else "./results"
        self.plots_dir = self.args.plots_dir if hasattr(self.args, 'plots_dir') else "./results/plots"
        self.log_dir = self.args.log_dir if hasattr(self.args, 'log_dir') else "./logs"
        
        # Compression
        self.compression_method = self.args.compression_method if hasattr(self.args, 'compression_method') else "standard_svd"
        self.compression_rank = self.args.compression_rank if hasattr(self.args, 'compression_rank') else 64
        self.compression_threshold = self.args.compression_threshold if hasattr(self.args, 'compression_threshold') else 0.90
        
        # Visualization
        self.visualize = self.args.visualize if hasattr(self.args, 'visualize') else True
        self.generate_report = self.args.generate_report if hasattr(self.args, 'generate_report') else True
        
        # Logging
        self.logging_steps = self.args.logging_steps if hasattr(self.args, 'logging_steps') else 50
        self.verbose = self.args.verbose if hasattr(self.args, 'verbose') else True
    
    def _update_from_dict(self, config: Dict):
        """Update configuration from dictionary."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """
    Main experiment runner for SVD-Lin framework.
    
    Handles:
    - Model loading and compression
    - Data loading and preprocessing
    - Training and evaluation
    - Hyperparameter optimization
    - Visualization and reporting
    """
    
    def __init__(self, config: MainConfig):
        self.config = config
        self.logger = None
        self.results = {}
        self.checkpoint_manager = CheckpointManager(os.path.join(config.output_dir, "checkpoints"))
        
    def setup(self) -> 'ExperimentRunner':
        """Setup experiment environment."""
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.config.log_dir, f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.logger = setup_logger(
            "svd_lin",
            level=logging.INFO if self.config.verbose else logging.WARNING,
            log_file=log_file
        )
        
        # Set seed for reproducibility
        set_seed(self.config.seed, deterministic=True)
        
        self.logger.info("=" * 60)
        self.logger.info("🚀 SVD-Lin Framework Initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Attention: {self.config.attention_type}")
        self.logger.info(f"Rank: {self.config.rank}")
        self.logger.info(f"Tasks: {self.config.tasks}")
        self.logger.info(f"Output Dir: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
        return self
    
    def load_model(self) -> Tuple[nn.Module, Any]:
        """Load and optionally compress model."""
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
        
        self.logger.info(f"\n📦 Loading model: {self.config.model_name}")
        
        # Detect architecture
        arch = detect_architecture(self.config.model_name)
        self.logger.info(f"   Architecture: {arch}")
        
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        config = AutoConfig.from_pretrained(self.config.model_name)
        
        # Determine number of labels for first task
        first_task = self.config.tasks[0]
        num_labels = get_num_labels(first_task)
        config.num_labels = num_labels
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, config=config
        )
        
        original_params = count_parameters(model)
        self.logger.info(f"   Original parameters: {format_number(original_params)}")
        
        # Apply compression if requested
        if self.config.attention_type != "full" or self.config.compression_method != "none":
            model = self._apply_compression(model)
        
        compressed_params = count_parameters(model)
        self.logger.info(f"   Compressed parameters: {format_number(compressed_params)}")
        self.logger.info(f"   Compression ratio: {compressed_params/original_params:.2%}")
        
        # Move to device
        model.to(self.config.device)
        
        return model, tokenizer
    
    def _apply_compression(self, model):
        """Apply compression to model."""
        self.logger.info(f"\n🔧 Applying compression: {self.config.compression_method}")
        
        if self.config.attention_type == "svd":
            # Replace attention with SVD attention
            from framework.core import replace_attention_in_model
            model = replace_attention_in_model(model, "svd", rank=self.config.rank)
            self.logger.info(f"   Replaced attention with SVD (rank={self.config.rank})")
        
        elif self.config.compression_method == "standard_svd":
            compressor = AdvancedZeroShotCompressor(
                method="standard_svd",
                base_threshold=self.config.compression_threshold,
                min_rank=4
            )
            model, stats, reduction = compressor.compress_model(model, "standard_svd")
            self.logger.info(f"   Standard SVD compression: {reduction:.1f}% reduction")
        
        elif self.config.compression_method == "adaptive_svd":
            rank_map = get_adaptive_ranks(self.config.model_name, ["query", "key", "value", "dense"])
            count = apply_adaptive_svd(model, rank_map)
            self.logger.info(f"   Adaptive SVD: compressed {count} layers")
        
        return model
    
    def load_data(self, task_name: str, tokenizer):
        """Load and preprocess data for a specific task."""
        self.logger.info(f"\n📂 Loading data for task: {task_name}")
        
        # Load GLUE task
        dataset = load_glue_task(task_name, self.config.local_glue_path)
        
        # Get task info
        is_regression = task_name == "stsb"
        num_labels = get_num_labels(task_name)
        
        # Preprocess
        preprocess_fn = preprocess_function_builder(
            tokenizer, task_name, self.config.max_seq_length, is_regression
        )
        
        num_proc = min(4, os.cpu_count() or 1)
        encoded = dataset.map(preprocess_fn, batched=True, num_proc=num_proc, remove_columns=["label"])
        
        # Set format
        cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in encoded["train"].column_names:
            cols.append("token_type_ids")
        encoded.set_format(type="torch", columns=cols)
        
        train_ds = encoded["train"]
        eval_key = "validation" if "validation" in encoded else "validation_matched"
        eval_ds = encoded[eval_key]
        
        self.logger.info(f"   Train samples: {len(train_ds)}")
        self.logger.info(f"   Validation samples: {len(eval_ds)}")
        
        return train_ds, eval_ds, num_labels, is_regression
    
    def train_model(self, model, train_ds, eval_ds, task_name: str, num_labels: int, is_regression: bool):
        """Train model on a specific task."""
        from transformers import TrainingArguments
        
        self.logger.info(f"\n🏋️ Training on {task_name}")
        
        # Calculate warmup steps
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps
        steps_per_epoch = len(train_ds) // effective_batch
        num_training_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        
        # Setup training arguments
        metric_name = get_metric_name(task_name)
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, f"{task_name}_checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_{metric_name}",
            greater_is_better=True,
            fp16=self.config.mixed_precision,
            dataloader_pin_memory=False,
            save_only_model=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(model.tokenizer if hasattr(model, 'tokenizer') else None)
        
        # Compute metrics function
        compute_metrics_fn = get_task_metrics(task_name)
        
        # Create trainer
        trainer = SVDAwareTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_metrics = trainer.evaluate()
        self.logger.info(f"   Results: {eval_metrics}")
        
        return eval_metrics, trainer
    
    def run_single_task(self, task_name: str, model, tokenizer) -> Dict[str, Any]:
        """Run complete experiment for a single task."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 Task: {task_name.upper()}")
        self.logger.info(f"{'='*60}")
        
        # Load data
        train_ds, eval_ds, num_labels, is_regression = self.load_data(task_name, tokenizer)
        
        # Train model
        eval_metrics, trainer = self.train_model(model, train_ds, eval_ds, task_name, num_labels, is_regression)
        
        # Benchmark (optional)
        benchmark_results = None
        if hasattr(self.config, 'benchmark') and self.config.benchmark:
            dataloader = DataLoader(eval_ds, batch_size=self.config.batch_size)
            benchmark_results = benchmark_model(model, dataloader, self.config.device)
        
        # Visualize
        if self.config.visualize:
            self._visualize_results(trainer, task_name, model)
        
        return {
            "task": task_name,
            "metrics": eval_metrics,
            "benchmark": benchmark_results,
            "num_params": count_parameters(model),
            "config": {
                "model": self.config.model_name,
                "attention": self.config.attention_type,
                "rank": self.config.rank,
                "compression": self.config.compression_method,
            }
        }
    
    def _visualize_results(self, trainer, task_name: str, model):
        """Generate visualizations for results."""
        self.logger.info(f"\n🎨 Generating visualizations for {task_name}")
        
        # Plot training curves
        if hasattr(trainer, 'state') and trainer.state.log_history:
            plot_training_curves(
                trainer.state.log_history,
                save_path=os.path.join(self.config.plots_dir, f"training_curves_{task_name}.png")
            )
        
        # Plot accuracy vs rank (if multiple ranks)
        # This would require aggregating results from multiple runs
    
    def run_all_tasks(self, model, tokenizer) -> Dict[str, Any]:
        """Run experiments for all configured tasks."""
        all_results = {}
        
        for task_name in self.config.tasks:
            try:
                result = self.run_single_task(task_name, model, tokenizer)
                all_results[task_name] = result
                
                # Clear memory between tasks
                clear_memory()
                
            except Exception as e:
                self.logger.error(f"❌ Failed on {task_name}: {e}")
                all_results[task_name] = {"error": str(e)}
        
        return all_results
    
    def run_hyperparameter_optimization(self, model, tokenizer) -> Dict[str, Any]:
        """Run hyperparameter optimization using Optuna."""
        self.logger.info("\n🔍 Running hyperparameter optimization")
        
        def train_fn(config):
            # Update config
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Run single task for optimization
            task_name = self.config.tasks[0]
            result = self.run_single_task(task_name, model, tokenizer)
            return result.get("metrics", {})
        
        def eval_fn(config):
            # Evaluate with given config
            return train_fn(config)
        
        config_template = {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'rank': 64,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
        }
        
        optimizer = HyperparameterOptimizer(train_fn, eval_fn, config_template)
        best_params = optimizer.run(n_trials=self.config.num_seeds * 10)
        
        self.logger.info(f"✅ Best parameters: {best_params}")
        return best_params
    
    def generate_final_report(self, results: Dict[str, Any]):
        """Generate final HTML report."""
        self.logger.info("\n📄 Generating final report")
        
        # Convert results to DataFrame
        rows = []
        for task, result in results.items():
            if "error" in result:
                continue
            metrics = result.get("metrics", {})
            row = {
                "task": task,
                "model": self.config.model_name,
                "rank": self.config.rank,
                "attention": self.config.attention_type,
            }
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    row[key.replace("eval_", "")] = value
            rows.append(row)
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            
            # Generate reports
            generate_csv_summary(
                {task: {self.config.rank: result.get("metrics", {})} for task, result in results.items()},
                os.path.join(self.config.output_dir, "summary.csv")
            )
            
            generate_html_report(
                self.config.output_dir,
                self.config.output_dir,
                df,
                title=f"SVD-Lin Report - {self.config.model_name}"
            )
            
            # Generate advanced report if Plotly available
            try:
                report_gen = AdvancedReportGenerator(self.config.output_dir)
                report_gen.generate_full_report(
                    results={},
                    df=df,
                    profiler_report=None,
                    layer_analysis=None
                )
            except Exception as e:
                self.logger.warning(f"Advanced report generation failed: {e}")
        
        # Save results as JSON
        save_json(results, os.path.join(self.config.output_dir, "all_results.json"))
        self.logger.info(f"✅ Results saved to {self.config.output_dir}")
    
    def run(self):
        """Main entry point for experiment runner."""
        try:
            # Setup
            self.setup()
            
            # Load model
            model, tokenizer = self.load_model()
            
            # Check if running hyperparameter optimization
            if hasattr(self.config, 'optimize') and self.config.optimize:
                best_params = self.run_hyperparameter_optimization(model, tokenizer)
                # Update config with best params
                for key, value in best_params.items():
                    setattr(self.config, key, value)
                self.logger.info(f"Updated config with best params: {best_params}")
            
            # Run experiments
            if self.config.tasks:
                results = self.run_all_tasks(model, tokenizer)
                self.results = results
                
                # Generate report
                if self.config.generate_report:
                    self.generate_final_report(results)
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("🎉 Experiment completed successfully!")
            self.logger.info(f"   Results saved to: {self.config.output_dir}")
            self.logger.info("=" * 60)
            
            return self.results
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Experiment interrupted by user")
            return self.results
        except Exception as e:
            self.logger.error(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SVD-Lin: Linearizing Transformer Attention via Truncated SVD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run baseline experiment
    python main.py --model bert-base-uncased --attention full --tasks cola sst2
    
    # Run SVD compression
    python main.py --model bert-base-uncased --attention svd --rank 64 --tasks cola mrpc
    
    # Run with compression method
    python main.py --model distilbert-base-uncased --compression_method standard_svd --compression_rank 32
    
    # Run hyperparameter optimization
    python main.py --model bert-base-uncased --optimize --num_seeds 5
    
    # Generate visualizations
    python main.py --model bert-base-uncased --visualize --generate_report
        """
    )
    
    # Model
    parser.add_argument("--model_name", "-m", type=str, default="bert-base-uncased",
                        help="HuggingFace model name")
    parser.add_argument("--attention_type", "-a", type=str, default="svd",
                        choices=["full", "svd", "nystrom", "performer", "linformer", "rala", "nha"],
                        help="Attention type")
    parser.add_argument("--rank", "-r", type=int, default=128,
                        help="SVD compression rank")
    
    # Data
    parser.add_argument("--tasks", "-t", nargs="+", default=["cola", "sst2", "mrpc"],
                        help="GLUE tasks to run")
    parser.add_argument("--local_glue_path", type=str, 
                        default="/Users/tahamajs/Documents/uni/Project/datasets/glue",
                        help="Path to local GLUE datasets")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Compression
    parser.add_argument("--compression_method", type=str, default="standard_svd",
                        choices=["none", "standard_svd", "adaptive_svd", "randomized_svd", "quantized_svd"],
                        help="Compression method")
    parser.add_argument("--compression_rank", type=int, default=64,
                        help="Rank for compression")
    parser.add_argument("--compression_threshold", type=float, default=0.90,
                        help="Energy threshold for compression")
    
    # Experiment
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds for multi-seed experiments")
    parser.add_argument("--optimize", action="store_true",
                        help="Run hyperparameter optimization")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use")
    
    # Output
    parser.add_argument("--output_dir", "-o", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--plots_dir", type=str, default="./results/plots",
                        help="Plots directory")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Configuration file path")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate HTML report")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging steps")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    if args.extend:
        plugin_dirs = ["framework/attentions/custom", "framework/compression/custom"]
        if args.plugin_dir:
            plugin_dirs.append(args.plugin_dir)
        discover_plugins(plugin_dirs)
        print(f"✅ Loaded custom plugins from: {plugin_dirs}")

    # List components if requested
    if args.list_components:
        components = list_all_components()
        print("\n📦 Available Components:")
        for comp_type, comp_list in components.items():
            print(f"\n  {comp_type.upper()}:")
            for comp in comp_list:
                print(f"    - {comp}")
        return


    # Create configuration
    config = MainConfig(args)
    
    # Create experiment runner and run
    runner = ExperimentRunner(config)
    results = runner.run()
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n✅ SVD-Lin execution completed!")