"""
Advanced interactive HTML report using Plotly dashboards.
Requires plotly.
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime


class AdvancedReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_rank_evolution_plot(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for task in df["task"].unique():
            sub = df[df["task"] == task]
            fig.add_trace(go.Scatter(x=sub["rank"], y=sub["accuracy"],
                                     mode="lines+markers", name=task))
        fig.update_layout(title="Performance vs Rank", xaxis_title="Rank", yaxis_title="Accuracy")
        return fig

    def create_model_comparison_plot(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            fig.add_trace(go.Scatter(x=sub["rank"], y=sub["accuracy"],
                                     mode="lines+markers", name=model))
        fig.update_layout(title="Model Comparison", xaxis_title="Rank", yaxis_title="Accuracy")
        return fig

    def generate_full_report(self, df: pd.DataFrame, output_path: str) -> str:
        rank_plot = self.create_rank_evolution_plot(df)
        model_plot = self.create_model_comparison_plot(df)
        html = f"""<!DOCTYPE html>
<html>
<head><title>SVD-Lin Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: Arial; margin: 20px; }}
    .plot {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1>Comprehensive Analysis Report</h1>
<p>Generated: {datetime.now()}</p>
<div class="plot"><div id="rankPlot"></div></div>
<div class="plot"><div id="modelPlot"></div></div>
<script>
    var rankData = {rank_plot.to_json()};
    Plotly.newPlot('rankPlot', rankData.data, rankData.layout);
    var modelData = {model_plot.to_json()};
    Plotly.newPlot('modelPlot', modelData.data, modelData.layout);
</script>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)
        return output_path