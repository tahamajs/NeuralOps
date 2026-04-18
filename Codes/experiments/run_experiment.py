import argparse
import yaml
from framework.core.registry import MODEL_REGISTRY, ATTENTION_REGISTRY, VISUALIZER_REGISTRY
from framework.models.model_loader import load_model
from framework.data.glue_loader import load_glue_task
from framework.training.trainer import SVDAwareTrainer
from framework.visualization.report_generator import generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--attention", type=str, default="svd", choices=ATTENTION_REGISTRY.keys())
    parser.add_argument("--rank", type=int, default=128, help="Rank for low‑rank attention")
    parser.add_argument("--task", type=str, required=True, help="GLUE task name")
    parser.add_argument("--visualizations", nargs="+", default=["attention_heatmap", "svd_spectrum"],
                        choices=VISUALIZER_REGISTRY.keys())
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    dataset = load_glue_task(args.task, local_path=config["local_glue_path"])

    # Load model with specified attention
    model, tokenizer = load_model(
        model_name=args.model,
        num_labels=dataset.num_labels,
        attention_type=args.attention,
        rank=args.rank if args.attention != "full" else None,
        device=config["device"]
    )

    # Train
    trainer = SVDAwareTrainer(
        model=model,
        train_dataset=dataset.train,
        eval_dataset=dataset.validation,
        tokenizer=tokenizer,
        config=config,
        visualizers=[VISUALIZER_REGISTRY.get(v)() for v in args.visualizations]
    )
    results = trainer.train()

    # Generate report
    generate_report(results, output_dir=config["output_dir"])

if __name__ == "__main__":
    main()