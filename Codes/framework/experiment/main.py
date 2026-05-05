#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning
Usage: python -m framework.experiment.main --attn_type nexus --method lora --task sst2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import torch
from framework.experiment.config import build_config_from_args
from framework.experiment.data import load_and_preprocess_data
from framework.experiment.model import load_model
from framework.experiment.train import setup_trainer
from experiment_logging.logger import setup_logging, MetricsTracker, LoggingConfig
from framework.utils.seed import set_all_seeds
from framework.utils.device import get_device

# Environment (offline, optional)
# os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
# os.environ["HF_HOME"] = "./models"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_training(config, stop_event=None, progress_callback=None, log_callback=None, final_callback=None):
    """
    API call for running training from the Django runner.
    """
    from framework.experiment.config import ExperimentConfig
    import types

    # Convert dict to namespace if needed
    if isinstance(config, dict):
        cfg = ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"
    
    # Simple direct setup
    set_all_seeds(cfg.seed)
    device = get_device()
    
    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)
    model = load_model(cfg, num_labels, device)
    model.to(device)

    # Use a minimal progress callback wrapper if needed
    # ... In a real scenario, we'd inject progress_callback into the trainer here ...
    
    trainer = setup_trainer(model, tokenizer, dataset, cfg, None, logging.getLogger("api"))
    
    # Execute
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    
    if final_callback:
        final_callback(eval_result, train_result.metrics if hasattr(train_result, 'metrics') else {})
    
    return eval_result

def run_evaluation(config, model_path):
    """Placeholder for evaluation runner."""
    return {"accuracy": 0.85, "loss": 0.4}

def run_compression(config, output_dir):
    """Placeholder for compression runner."""
    return {"reduction": "4x", "original_size_mb": 500, "new_size_mb": 125}

def main():
    cfg = build_config_from_args()

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"
    # Ensure necessary attributes exist for the logger
    if not hasattr(cfg, 'log_base_dir'):
        cfg.log_base_dir = cfg.log_dir
    if not hasattr(cfg, 'use_timestamp'):
        cfg.use_timestamp = True
    if not hasattr(cfg, 'run_id'):
        cfg.run_id = None

    # Build LoggingConfig from ExperimentConfig
    log_cfg = LoggingConfig(
        log_base_dir=cfg.log_base_dir,
        experiment_name=exp_name,
        run_id=cfg.run_id,
        use_timestamp=cfg.use_timestamp,
    )
    logger = setup_logging(log_cfg)

    tracker = MetricsTracker(save_path=os.path.join(cfg.log_dir, f"{exp_name}_metrics.json"))

    logger.info("══════════════════════════════════════")
    logger.info(f"🚀 Experiment: {exp_name}")
    logger.info(f"   Attention: {cfg.attn_type}")
    logger.info(f"   Method: {cfg.method}")
    logger.info(f"   Task: {cfg.task}")
    logger.info(f"   Model: {cfg.model_name}")
    logger.info("══════════════════════════════════════")

    set_all_seeds(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    # Model
    model = load_model(cfg, num_labels, device)
    model.to(device)

    # Trainer
    trainer = setup_trainer(model, tokenizer, dataset, cfg, tracker, logger)

    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    # Save final model and logs
    output_dir = os.path.join(cfg.output_dir, exp_name)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.json"))
    logger.info(f"Model and logs saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()