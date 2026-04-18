from ..core.registry import VISUALIZER_REGISTRY
from ..core.base_visualizer import BaseVisualizer

@VISUALIZER_REGISTRY.register("attention_heatmap")
class AttentionHeatmapVisualizer(BaseVisualizer):
    def generate(self, attention_matrix, **kwargs):
        # ... plotting logic
        return figure_path