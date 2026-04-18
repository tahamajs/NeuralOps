import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from visualization import SVDAttentionVisualizer
from transformers import AutoTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, DistilBertForSequenceClassification
from modeling_bert_svd import BertModelSVD
import argparse

def load_model(model_name, task, rank, num_labels):
    """Load a trained model with SVD attention."""
    from transformers import BertConfig
    
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
    if rank is None or rank == 'None':
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    else:
        model = BertForSequenceClassification(config)
        model.bert = BertModelSVD(config, rank=int(rank))
        # Load trained weights
        checkpoint_path = f"results/run_*/{task}/rank_{rank}/checkpoint-*"
        # (implement checkpoint loading logic)
    
    return model

def analyze_svd_compression_quality(attention_matrix, rank):
    """Analyze how well SVD compression preserves attention information."""
    U, s, Vt = np.linalg.svd(attention_matrix, full_matrices=False)
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    
    # Energy retained at rank
    energy_retained = cumulative_energy[min(rank, len(cumulative_energy)-1)]
    
    # Optimal rank for 95% energy
    optimal_rank = np.argmax(cumulative_energy >= 0.95) + 1
    
    return {
        'energy_retained': energy_retained,
        'optimal_rank': optimal_rank,
        'singular_values': s.tolist(),
        'cumulative_energy': cumulative_energy.tolist()
    }

def generate_full_report(results_dir, output_dir):
    """Generate a comprehensive HTML report of all visualizations."""
    import json
    from datetime import datetime
    
    visualizer = SVDAttentionVisualizer(output_dir=output_dir)
    
    # Load all results
    all_results = {}
    for run_dir in os.listdir(results_dir):
        if run_dir.startswith('run_'):
            run_path = os.path.join(results_dir, run_dir)
            results_file = os.path.join(run_path, 'all_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    all_results[run_dir] = json.load(f)
    
    # Generate summary plots
    # ... (implementation)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SVD-Lin Attention Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .card {{ margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>SVD-Lin Attention Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Experiment Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Models</td><td>BERT, RoBERTa, DistilBERT</td></tr>
            <tr><td>Ranks</td><td>64, 128, 256</td></tr>
            <tr><td>Tasks</td><td>GLUE (CoLA, SST-2, MRPC, etc.)</td></tr>
        </table>
        
        <h2>Visualizations</h2>
        <div class="container">
            <!-- Embed all generated images -->
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_content)
    
    return html_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="analysis_output")
    args = parser.parse_args()
    
    generate_full_report(args.results_dir, args.output_dir)
    print(f"✅ Report generated at {args.output_dir}/report.html")