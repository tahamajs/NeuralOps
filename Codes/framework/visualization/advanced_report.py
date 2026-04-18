import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class AdvancedReportGenerator:
    """Generate interactive HTML reports with Plotly."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_attention_heatmap_interactive(self, attention_matrix, tokens, layer, head):
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
            height=800
        )
        return fig
    
    def create_rank_evolution_plot(self, results_df):
        """Interactive plot of accuracy vs rank for multiple tasks."""
        fig = go.Figure()
        for task in results_df['task'].unique():
            df_task = results_df[results_df['task'] == task]
            fig.add_trace(go.Scatter(
                x=df_task['rank'],
                y=df_task['accuracy'],
                mode='lines+markers',
                name=task
            ))
        fig.update_layout(
            title='Performance vs Compression Rank',
            xaxis_title='Rank',
            yaxis_title='Accuracy',
            hovermode='closest'
        )
        return fig
    
    def generate_full_report(self, results, df, profiler_report, layer_analysis):
        """Generate complete HTML with tabs and interactive graphs."""
        # Create Plotly figures
        rank_fig = self.create_rank_evolution_plot(df)
        # ... more figures
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVD-Lin Advanced Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
                .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                .tab button:hover {{ background-color: #ddd; }}
                .tab button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
            </style>
        </head>
        <body>
            <h1>SVD-Lin Comprehensive Analysis Report</h1>
            <div class="tab">
                <button class="tablinks" onclick="openTab(event, 'Performance')">Performance Summary</button>
                <button class="tablinks" onclick="openTab(event, 'Profiling')">Profiling</button>
                <button class="tablinks" onclick="openTab(event, 'LayerAnalysis')">Layer Analysis</button>
                <button class="tablinks" onclick="openTab(event, 'AttentionMaps')">Attention Maps</button>
            </div>
            
            <div id="Performance" class="tabcontent">
                <h2>Performance Metrics</h2>
                {df.to_html()}
                <div id="rank_plot"></div>
            </div>
            
            <div id="Profiling" class="tabcontent">
                <h2>Profiling Report</h2>
                <pre>{json.dumps(profiler_report, indent=2)}</pre>
            </div>
            
            <div id="LayerAnalysis" class="tabcontent">
                <h2>Layer-wise Analysis</h2>
                <pre>{json.dumps(layer_analysis, indent=2)}</pre>
            </div>
            
            <div id="AttentionMaps" class="tabcontent">
                <h2>Attention Visualizations</h2>
                <!-- Embed interactive plots -->
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
            </script>
        </body>
        </html>
        """
        with open(os.path.join(self.output_dir, 'advanced_report.html'), 'w') as f:
            f.write(html)