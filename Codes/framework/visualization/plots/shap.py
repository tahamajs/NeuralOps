"""
SHAP explanation – local interpretability via SHAP values for the model on a sample.
Requires shap to be installed. If not, logs a warning and skips.
"""
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_shap(history, metrics, config, output_dir, opts):
    try:
        import shap
    except ImportError:
        logger.info("SHAP plot requested but shap not installed. Run: pip install shap")
        return []

    # We need the model and tokenizer. They may be stored in the run's artifacts.
    model_path = Path(output_dir).parent / "artifacts" / "model"
    tokenizer_path = model_path / "tokenizer"
    if not (model_path.exists() and tokenizer_path.exists()):
        logger.info("SHAP plot requires a saved model & tokenizer. Skipping.")
        return []

    # Load a sample of evaluation texts
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("SHAP requires eval_predictions.json with text samples.")
        return []
    with open(preds_path) as f:
        data = json.load(f)
    texts = data.get("texts", [])
    if len(texts) == 0:
        return []

    # Load model and tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception as e:
        logger.exception(f"Failed to load model for SHAP: {e}")
        return []

    # Pick a few samples
    n_samples = min(opts.get("shap_n_samples", 2), len(texts))
    sample_texts = texts[:n_samples]

    try:
        # Build SHAP explainer (simple gradient‑based)
        explainer = shap.Explainer(model, tokenizer)
        shap_values = explainer(sample_texts)
    except Exception as e:
        logger.exception(f"SHAP explainer failed: {e}")
        return []

    saved = []
    for i in range(n_samples):
        fig = plt.figure(figsize=opts.get("figsize", [12, 4]))
        shap.plots.text(shap_values[i], display=False)
        out_path = str(Path(output_dir) / f"shap_sample_{i}.png")
        fig.savefig(out_path, dpi=opts.get("dpi", 150), bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)

    return saved