#!/usr/bin/env python
# studio/cli.py
"""
Command-line interface for SVD-Lin Studio.
"""

import argparse
import sys
import json
from pathlib import Path

from studio.core.model_manager import ModelManager
from studio.core.attention_manager import AttentionManager
from studio.core.compression_manager import CompressionManager
from studio.core.visualization_manager import VisualizationManager


def main():
    parser = argparse.ArgumentParser(
        description="SVD-Lin Studio - Transformer Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load and analyze model
  studio load bert-base-uncased --info
  
  # Apply SVD attention
  studio compress bert-base-uncased --attention svd --rank 64
  
  # Visualize attention
  studio visualize bert-base-uncased --text "Hello world" --layer 0 --head 0
  
  # Run full pipeline
  studio pipeline --model bert-base-uncased --task cola --rank 64
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('model', type=str, help='Model name')
    load_parser.add_argument('--task', type=str, default='sequence_classification')
    load_parser.add_argument('--info', action='store_true', help='Show model info')
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress model')
    compress_parser.add_argument('model', type=str, help='Model name')
    compress_parser.add_argument('--attention', type=str, default='svd', help='Attention type')
    compress_parser.add_argument('--rank', type=int, default=64, help='Compression rank')
    compress_parser.add_argument('--output', type=str, help='Output path')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize attention')
    viz_parser.add_argument('model', type=str, help='Model name')
    viz_parser.add_argument('--text', type=str, default='The cat sat on the mat')
    viz_parser.add_argument('--layer', type=int, default=0, help='Layer index')
    viz_parser.add_argument('--head', type=int, default=0, help='Head index')
    viz_parser.add_argument('--output', type=str, help='Output file')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--model', type=str, default='bert-base-uncased')
    pipeline_parser.add_argument('--task', type=str, default='cola')
    pipeline_parser.add_argument('--rank', type=int, default=64)
    pipeline_parser.add_argument('--epochs', type=int, default=3)
    
    # List models
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--pattern', type=str, default='*', help='Search pattern')
    
    args = parser.parse_args()
    
    if args.command == 'load':
        manager = ModelManager()
        model, tokenizer = manager.load_model(args.model, task=args.task)
        
        print(f"✅ Loaded {args.model}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if args.info:
            info = manager.get_model_summary(model)
            print(json.dumps(info, indent=2))
    
    elif args.command == 'compress':
        manager = ModelManager()
        model, tokenizer = manager.load_model(args.model)
        
        attn_manager = AttentionManager()
        model = attn_manager.apply_attention(model, args.attention, rank=args.rank)
        
        print(f"✅ Applied {args.attention} attention (rank={args.rank})")
        
        if args.output:
            manager.save_model(model, tokenizer, args.output)
            print(f"   Saved to {args.output}")
    
    elif args.command == 'visualize':
        manager = ModelManager()
        model, tokenizer = manager.load_model(args.model)
        
        inputs = tokenizer(args.text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        if hasattr(outputs, 'attentions'):
            attn = outputs.attentions[args.layer][0, args.head]
            vis = VisualizationManager()
            fig = vis.plot_attention_heatmap(attn, interactive=True)
            
            if args.output:
                fig.write_html(args.output)
                print(f"✅ Saved visualization to {args.output}")
            else:
                fig.show()
        else:
            print("Model does not output attention maps")
    
    elif args.command == 'pipeline':
        # Run full training pipeline
        from studio.training import Trainer
        trainer = Trainer(
            model_name=args.model,
            task=args.task,
            rank=args.rank,
            epochs=args.epochs
        )
        results = trainer.run()
        print(json.dumps(results, indent=2))
    
    elif args.command == 'list':
        from huggingface_hub import list_models
        models = list_models(filter=args.pattern, limit=20)
        print(f"Available models (showing first 20):")
        for model in models:
            print(f"  - {model.modelId}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()