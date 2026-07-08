import logging

logger = logging.getLogger(__name__)


def plot_attention_heatmap(history, metrics, config, output_dir, opts):
    # This plot requires the actual model and a data sample – not available from history alone.
    # If you saved an attention matrix during training, you can load it here.
    attn_path = output_dir.parent / "artifacts" / "attention_sample.json"
    if not attn_path.exists():
        logger.info(
            "Attention heatmap requested but no attention data saved. "
            "To enable this, log an attention matrix during training."
        )
        return []
    import json

    import numpy as np

    with open(attn_path) as f:
        attn_data = json.load(f)
    matrix = np.array(attn_data.get("matrix", []))
    tokens = attn_data.get("tokens", None)
    layer = attn_data.get("layer", 0)
    head = attn_data.get("head", 0)

    from framework.visualization.svd_analysis import SVDAttentionVisualizer

    viz = SVDAttentionVisualizer(str(output_dir))
    path = viz.plot_attention_heatmap(
        matrix,
        tokens=tokens,
        layer_idx=layer,
        head_idx=head,
        rank=config.get("compression_rank", 0),
        model_name=config.get("model_name", "model"),
        task_name=config.get("task_name", "task"),
    )
    return [path] if path else []
