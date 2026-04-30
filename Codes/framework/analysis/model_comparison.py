# framework/analysis/model_comparison.py
"""
Model comparison utilities for SVD compression analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import numpy as np


class ModelComparison:
    """
    Compare multiple models across different compression ranks and GLUE tasks.
    
    Usage:
        comparator = ModelComparison(results_dirs=['./results/bert_run', './results/roberta_run'])
        comparator.load_all_results()
        comparator.plot_accuracy_vs_rank(task='cola')
        summary = comparator.generate_summary_table()
    """
    
    def __init__(self, results_dirs: List[str]):
        """
        Args:
            results_dirs: List of directories containing experiment results
                         (each should have performance.csv and/or summary.json)
        """
        self.results_dirs = results_dirs
        self.dataframes = []
        self.df_all = None
        self.all_results = {}
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all results from CSV and JSON files in the given directories."""
        for d in self.results_dirs:
            run_name = os.path.basename(d)
            
            # Try to load from performance.csv first
            csv_path = os.path.join(d, 'performance.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['run'] = run_name
                
                # Extract model name from run name or config
                if 'model' not in df.columns:
                    df['model'] = self._extract_model_name(d)
                self.dataframes.append(df)
                self.all_results[run_name] = {'csv': df}
            
            # Also load summary.json if exists
            json_path = os.path.join(d, 'summary_all_tasks.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    summary = json.load(f)
                self.all_results[run_name]['json'] = summary
                
            # Load individual task summaries
            task_dir = Path(d)
            for task_subdir in task_dir.iterdir():
                if task_subdir.is_dir():
                    task_json = task_subdir / 'summary.json'
                    if task_json.exists():
                        with open(task_json, 'r') as f:
                            task_data = json.load(f)
                        if 'json_tasks' not in self.all_results[run_name]:
                            self.all_results[run_name]['json_tasks'] = {}
                        self.all_results[run_name]['json_tasks'][task_subdir.name] = task_data
        
        if self.dataframes:
            self.df_all = pd.concat(self.dataframes, ignore_index=True)
            self._add_derived_columns()
        else:
            # If no CSV files, construct from JSON
            self._build_from_json()
            
        return self.df_all
    
    def _extract_model_name(self, dir_path: str) -> str:
        """Extract model name from directory path or config."""
        config_path = os.path.join(dir_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('model_name_or_path', 'unknown')
        return os.path.basename(dir_path).split('_')[0]
    
    def _add_derived_columns(self):
        """Add computed columns like accuracy_drop, param_reduction, etc."""
        if 'original_accuracy' in self.df_all.columns:
            self.df_all['accuracy_drop'] = self.df_all['original_accuracy'] - self.df_all['compressed_accuracy']
        elif 'accuracy' in self.df_all.columns and 'rank' in self.df_all.columns:
            # Calculate drop relative to full attention
            full_accuracy = self.df_all[self.df_all['rank'] == self.df_all['rank'].max()]['accuracy'].mean()
            self.df_all['accuracy_drop'] = full_accuracy - self.df_all['accuracy']
        
        # Add compression ratio if not present
        if 'param_reduction' not in self.df_all.columns and 'original_params' in self.df_all.columns:
            self.df_all['param_reduction'] = (self.df_all['original_params'] - self.df_all['compressed_params']) / self.df_all['original_params']
            self.df_all['param_reduction'] = self.df_all['param_reduction'] * 100  # Convert to percentage
    
    def _build_from_json(self):
        """Build dataframe from JSON summaries when CSV not available."""
        rows = []
        for run_name, data in self.all_results.items():
            if 'json_tasks' in data:
                for task, task_results in data['json_tasks'].items():
                    for rank, metrics in task_results.items():
                        if rank == 'error':
                            continue
                        row = {
                            'run': run_name,
                            'model': self._extract_model_name(run_name),
                            'task': task,
                            'rank': int(rank) if rank != 'None' else 0,
                        }
                        # Extract metrics
                        for k, v in metrics.items():
                            if k.startswith('eval_'):
                                row[k.replace('eval_', '')] = v
                        rows.append(row)
        if rows:
            self.df_all = pd.DataFrame(rows)
            self._add_derived_columns()
    
    def plot_accuracy_vs_rank(self, task: str = 'cola', save: bool = True, 
                               show: bool = False, save_path: str = None) -> plt.Figure:
        """
        Plot accuracy vs compression rank for a specific task.
        
        Args:
            task: GLUE task name (cola, mrpc, sst2, etc.)
            save: Whether to save the figure to disk
            show: Whether to display the figure
            save_path: Custom save path for the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.df_all is None:
            self.load_all_results()
        
        df_task = self.df_all[self.df_all['task'] == task]
        if df_task.empty:
            print(f"Warning: No data found for task '{task}'")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines for each model
        for model in df_task['model'].unique():
            model_data = df_task[df_task['model'] == model]
            # Sort by rank
            model_data = model_data.sort_values('rank')
            ax.plot(model_data['rank'], model_data['accuracy'], 
                    marker='o', label=model, linewidth=2, markersize=8)
        
        ax.set_xlabel('SVD Rank', fontsize=12)
        ax.set_ylabel('Accuracy / Score', fontsize=12)
        ax.set_title(f'Performance vs Compression Rank - {task.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save:
            save_file = save_path or f'comparison_{task}.png'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Saved figure: {save_file}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_all_tasks(self, save_dir: str = './comparison_plots') -> Dict[str, plt.Figure]:
        """Generate accuracy vs rank plots for all tasks."""
        if self.df_all is None:
            self.load_all_results()
        
        os.makedirs(save_dir, exist_ok=True)
        figures = {}
        
        for task in self.df_all['task'].unique():
            fig = self.plot_accuracy_vs_rank(task=task, save=False)
            if fig:
                fig.savefig(os.path.join(save_dir, f'comparison_{task}.png'), dpi=150, bbox_inches='tight')
                figures[task] = fig
            plt.close()
        
        print(f"Plots saved to {save_dir}")
        return figures
    
    def generate_summary_table(self, save_path: str = 'summary_table.csv') -> pd.DataFrame:
        """Generate a pivot table summary of results."""
        if self.df_all is None:
            self.load_all_results()
        
        # Determine which metric to use
        metric_col = None
        for col in ['accuracy', 'f1', 'matthews_correlation', 'pearson_spearman']:
            if col in self.df_all.columns:
                metric_col = col
                break
        
        if metric_col is None:
            print("Warning: No metric column found")
            return pd.DataFrame()
        
        # Create pivot table
        pivot = self.df_all.pivot_table(
            index=['model', 'rank'], 
            columns='task', 
            values=metric_col,
            aggfunc='first'
        )
        
        # Save to CSV
        pivot.to_csv(save_path)
        print(f"Saved: {save_path}")
        
        return pivot
    
    def generate_model_ranking(self, metric: str = 'accuracy') -> pd.DataFrame:
        """Generate ranking of models by average performance."""
        if self.df_all is None:
            self.load_all_results()
        
        if metric not in self.df_all.columns:
            print(f"Warning: Metric '{metric}' not found")
            return pd.DataFrame()
        
        ranking = self.df_all.groupby('model')[metric].mean().sort_values(ascending=False).reset_index()
        ranking.columns = ['model', f'avg_{metric}']
        
        return ranking
    
    def get_best_config_per_task(self) -> pd.DataFrame:
        """Get best configuration for each task."""
        if self.df_all is None:
            self.load_all_results()
        
        best_configs = []
        for task in self.df_all['task'].unique():
            df_task = self.df_all[self.df_all['task'] == task]
            if 'accuracy' in df_task.columns:
                best_idx = df_task['accuracy'].idxmax()
                best_configs.append(df_task.loc[best_idx])
        
        return pd.DataFrame(best_configs)
    
    def _extract_method(self, row: pd.Series) -> str:
        """Extract compression method from row data."""
        if 'method' in row:
            return row['method']
        if 'run' in row:
            if 'svd' in str(row['run']).lower():
                return 'svd'
            if 'lora' in str(row['run']).lower():
                return 'lora'
        return 'unknown'


def quick_compare(results_dirs: List[str], task: str = 'cola', save_dir: str = './comparison'):
    """
    Quick comparison function for one-line usage.
    
    Args:
        results_dirs: List of result directories
        task: GLUE task to analyze
        save_dir: Directory to save outputs
        
    Returns:
        Summary table DataFrame
    """
    os.makedirs(save_dir, exist_ok=True)
    
    comparator = ModelComparison(results_dirs)
    comparator.load_all_results()
    comparator.plot_accuracy_vs_rank(task=task, save=True, save_path=os.path.join(save_dir, f'comparison_{task}.png'))
    
    summary = comparator.generate_summary_table(save_path=os.path.join(save_dir, 'summary_table.csv'))
    
    return summary