# framework/visualization/advanced_report.py
"""
Advanced interactive HTML report generation using Plotly.
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List


class AdvancedReportGenerator:
    """Generate interactive HTML reports with Plotly."""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save the report
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_attention_heatmap_interactive(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        layer: int,
        head: int
    ) -> go.Figure:
        """
        Create interactive attention heatmap.
        
        Args:
            attention_matrix: Attention weights matrix
            tokens: Token labels
            layer: Layer index
            head: Head index
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Attention Heatmap - Layer {layer}, Head {head}',
            width=800,
            height=800,
            xaxis_title='Key Tokens',
            yaxis_title='Query Tokens'
        )
        
        return fig
    
    def create_rank_evolution_plot(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create interactive plot of accuracy vs rank for multiple tasks.
        
        Args:
            results_df: DataFrame with 'task', 'rank', 'accuracy' columns
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for task in results_df['task'].unique():
            df_task = results_df[results_df['task'] == task]
            fig.add_trace(go.Scatter(
                x=df_task['rank'],
                y=df_task['accuracy'],
                mode='lines+markers',
                name=task,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Performance vs Compression Rank',
            xaxis_title='SVD Rank',
            yaxis_title='Accuracy / Score',
            hovermode='closest',
            legend_title='Task',
            template='plotly_white'
        )
        
        return fig
    
    def create_model_comparison_plot(
        self,
        results_df: pd.DataFrame,
        task: str = 'cola'
    ) -> go.Figure:
        """
        Create interactive model comparison plot.
        
        Args:
            results_df: DataFrame with 'model', 'rank', 'accuracy' columns
            task: Specific task to compare
            
        Returns:
            Plotly figure
        """
        df_task = results_df[results_df['task'] == task]
        
        fig = go.Figure()
        
        for model in df_task['model'].unique():
            df_model = df_task[df_task['model'] == model]
            fig.add_trace(go.Scatter(
                x=df_model['rank'],
                y=df_model['accuracy'],
                mode='lines+markers',
                name=model,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'Model Comparison - {task.upper()}',
            xaxis_title='SVD Rank',
            yaxis_title='Accuracy',
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def create_speedup_accuracy_tradeoff(
        self,
        results_df: pd.DataFrame
    ) -> go.Figure:
        """
        Create tradeoff plot between speedup and accuracy.
        
        Args:
            results_df: DataFrame with 'model', 'rank', 'accuracy', 'speedup_factor'
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Get full attention baseline
        full_acc = results_df[results_df['rank'] == results_df['rank'].max()]['accuracy'].mean()
        
        for model in results_df['model'].unique():
            df_model = results_df[results_df['model'] == model]
            
            # Calculate accuracy drop
            acc_drop = full_acc - df_model['accuracy']
            
            fig.add_trace(go.Scatter(
                x=df_model['speedup_factor'],
                y=acc_drop,
                mode='markers+text',
                name=model,
                text=df_model['rank'].astype(str),
                textposition='top center',
                marker=dict(size=12)
            ))
        
        fig.update_layout(
            title='Accuracy-Speedup Tradeoff',
            xaxis_title='Speedup Factor',
            yaxis_title='Accuracy Drop',
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=1, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_compression_heatmap(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Create compression ratio heatmap.
        
        Args:
            results_df: DataFrame with 'model', 'rank', 'compression_ratio'
            
        Returns:
            Plotly figure
        """
        pivot = results_df.pivot_table(
            index='model',
            columns='rank',
            values='compression_ratio',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Compression Ratio Heatmap',
            xaxis_title='SVD Rank',
            yaxis_title='Model',
            template='plotly_white'
        )
        
        return fig
    
    def generate_full_report(
        self,
        results: Dict,
        df: pd.DataFrame,
        profiler_report: Optional[Dict] = None,
        layer_analysis: Optional[Dict] = None,
        attention_maps: Optional[List] = None
    ) -> str:
        """
        Generate complete HTML report with tabs and interactive graphs.
        
        Args:
            results: Results dictionary
            df: DataFrame with performance metrics
            profiler_report: Profiling report
            layer_analysis: Layer-wise analysis results
            attention_maps: List of attention map figures
            
        Returns:
            Path to generated HTML file
        """
        # Create Plotly figures
        rank_fig = self.create_rank_evolution_plot(df)
        model_comparison_fig = self.create_model_comparison_plot(df)
        
        if 'speedup_factor' in df.columns:
            tradeoff_fig = self.create_speedup_accuracy_tradeoff(df)
        
        if 'compression_ratio' in df.columns:
            heatmap_fig = self.create_compression_heatmap(df)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SVD-Lin Comprehensive Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 16px; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #3498db; color: white; }}
        .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background-color: white; border-radius: 0 0 5px 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric-card {{ display: inline-block; background-color: #ecf0f1; padding: 15px; margin: 10px; border-radius: 10px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .plot-container {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SVD-Lin Comprehensive Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="tab">
        <button class="tablinks" onclick="openTab(event, 'Performance')" id="defaultOpen">Performance Summary</button>
        <button class="tablinks" onclick="openTab(event, 'Profiling')">Profiling Analysis</button>
        <button class="tablinks" onclick="openTab(event, 'LayerAnalysis')">Layer Analysis</button>
        <button class="tablinks" onclick="openTab(event, 'AttentionMaps')">Attention Maps</button>
        <button class="tablinks" onclick="openTab(event, 'Tradeoffs')">Tradeoffs</button>
    </div>
    
    <div id="Performance" class="tabcontent">
        <h2>Performance Metrics</h2>
        {df.to_html(index=False)}
        
        <div class="plot-container">
            <h3>Accuracy vs Rank</h3>
            <div id="rank_plot"></div>
        </div>
        
        <div class="plot-container">
            <h3>Model Comparison</h3>
            <div id="model_comparison_plot"></div>
        </div>
    </div>
    
    <div id="Profiling" class="tabcontent">
        <h2>Profiling Report</h2>
        <pre>{json.dumps(profiler_report or {}, indent=2)}</pre>
    </div>
    
    <div id="LayerAnalysis" class="tabcontent">
        <h2>Layer-wise Analysis</h2>
        <pre>{json.dumps(layer_analysis or {}, indent=2)}</pre>
    </div>
    
    <div id="AttentionMaps" class="tabcontent">
        <h2>Attention Visualizations</h2>
        <div id="attention_plots"></div>
    </div>
    
    <div id="Tradeoffs" class="tabcontent">
        <h2>Accuracy-Speedup Tradeoffs</h2>
        <div id="tradeoff_plot"></div>
        <div id="heatmap_plot"></div>
    </div>
    
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Embed Plotly figures
        var rankPlot = {rank_fig.to_json()};
        Plotly.newPlot('rank_plot', rankPlot.data, rankPlot.layout);
        
        var modelComparisonPlot = {model_comparison_fig.to_json()};
        Plotly.newPlot('model_comparison_plot', modelComparisonPlot.data, modelComparisonPlot.layout);
        
        var tradeoffPlot = {tradeoff_fig.to_json() if 'tradeoff_fig' in dir() else 'null'};
        if (tradeoffPlot) Plotly.newPlot('tradeoff_plot', tradeoffPlot.data, tradeoffPlot.layout);
        
        var heatmapPlot = {heatmap_fig.to_json() if 'heatmap_fig' in dir() else 'null'};
        if (heatmapPlot) Plotly.newPlot('heatmap_plot', heatmapPlot.data, heatmapPlot.layout);
        
        // Open default tab
        document.getElementById("defaultOpen").click();
    </script>
</body>
</html>"""
        
        output_path = os.path.join(self.output_dir, 'advanced_report.html')
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✅ Advanced report saved to {output_path}")
        return output_path