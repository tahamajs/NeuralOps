#!/usr/bin/env python
# examples/quick_start.py
"""
Quick start example for using SVD-Lin framework.
Shows how to easily compress any transformer model.
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Import the framework
from framework.quick import quick_svd_compress, quick_inference
from framework.core.model_wrapper import UnifiedTransformerModel


def example_1_basic_compression():
    """Example 1: Basic model compression."""
    print("\n" + "="*60)
    print("Example 1: Basic Model Compression")
    print("="*60)
    
    # One line to load and compress
    model = quick_svd_compress(
        model_name="bert-base-uncased",
        rank=64,
        attention_type="svd",
        num_labels=2
    )
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Test inference
    text = "This movie is absolutely amazing!"
    result = quick_inference(model, text, tokenizer)
    
    print(f"\nInput: {text}")
    print(f"Output logits shape: {result.logits.shape}")
    print(f"Prediction: {torch.argmax(result.logits, dim=-1).item()}")
    
    return model


def example_2_custom_attention():
    """Example 2: Using different attention types."""
    print("\n" + "="*60)
    print("Example 2: Different Attention Types")
    print("="*60)
    
    attention_types = ["svd", "nystrom", "performer"]
    ranks = [32, 64, 128]
    
    for attn_type in attention_types:
        for rank in ranks:
            print(f"\nTesting {attn_type} attention with rank={rank}...")
            
            model = quick_svd_compress(
                model_name="distilbert-base-uncased",
                rank=rank,
                attention_type=attn_type
            )
            
            print(f"  ✅ Success! Parameters: {model.get_parameter_count()}")


def example_3_fine_tuning():
    """Example 3: Fine-tune compressed model on GLUE."""
    print("\n" + "="*60)
    print("Example 3: Fine-tuning on GLUE")
    print("="*60)
    
    # Load and compress model
    model = quick_svd_compress(
        model_name="bert-base-uncased",
        rank=32,
        attention_type="svd",
        num_labels=2
    )
    
    # Load GLUE dataset
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Fine-tune (just a few epochs for demo)
    from framework.trainer import create_trainer
    
    trainer = create_trainer(
        model=model.model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        output_dir="./output_quick",
        batch_size=16,
        epochs=1,  # Just for demo
        learning_rate=2e-5,
    )
    
    trainer.train()
    results = trainer.evaluate()
    print(f"\n✅ Evaluation results: {results}")


def example_4_custom_pipeline():
    """Example 4: Custom compression pipeline."""
    print("\n" + "="*60)
    print("Example 4: Custom Compression Pipeline")
    print("="*60)
    
    # Load model with wrapper
    model = UnifiedTransformerModel.from_pretrained(
        "bert-base-uncased",
        task="sequence_classification",
        num_labels=2
    )
    
    print(f"\nOriginal parameters: {model.get_parameter_count()}")
    
    # Replace attention with SVD
    model.replace_attention("svd", rank=64)
    print(f"After attention replacement: {model.get_parameter_count()}")
    
    # Compress linear layers
    model.compress_linear_layers(rank=32)
    print(f"After linear compression: {model.get_parameter_count()}")
    
    # Apply LoRA
    model.apply_petl(method="lora", rank=8)
    print(f"After LoRA: {model.get_parameter_count()}")
    
    # Freeze early layers
    model.freeze_layers(num_layers_to_freeze=6)
    print(f"After freezing layers: {model.get_parameter_count()}")
    
    return model


def example_5_multi_model_comparison():
    """Example 5: Compare different models."""
    print("\n" + "="*60)
    print("Example 5: Model Comparison")
    print("="*60)
    
    models = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
    ranks = [32, 64, 128]
    
    results = []
    
    for model_name in models:
        for rank in ranks:
            print(f"\nCompressing {model_name} with rank {rank}...")
            
            model = quick_svd_compress(model_name, rank=rank)
            param_count = model.get_parameter_count()
            
            results.append({
                "model": model_name,
                "rank": rank,
                "total_params": param_count["total"],
                "trainable_params": param_count["trainable"],
            })
    
    # Print summary
    print("\n" + "-"*60)
    print("Summary:")
    print("-"*60)
    for r in results:
        print(f"  {r['model']} (rank={r['rank']}): {r['total_params']:,} params")
    
    return results


if __name__ == "__main__":
    print("🚀 SVD-Lin Quick Start Examples")
    print("="*60)
    
    # Run examples (uncomment to run)
    # example_1_basic_compression()
    # example_2_custom_attention()
    # example_3_fine_tuning()
    # example_4_custom_pipeline()
    # example_5_multi_model_comparison()
    
    print("\n✅ Quick start complete!")