"""Real‑time LLM quality and safety monitoring."""

from .models import LLMMonitorAlert


def evaluate_llm_output(output_text: str, reference_text: str = None) -> dict:
    """Compute quality metrics for an LLM output."""
    metrics = {}

    # ROUGE (simplified)
    if reference_text:
        from rouge import Rouge

        rouge = Rouge()
        scores = rouge.get_scores(output_text, reference_text)
        metrics["rouge-1"] = scores[0]["rouge-1"]["f"]
        metrics["rouge-l"] = scores[0]["rouge-l"]["f"]

    # BERTScore
    try:
        from bert_score import score

        P, R, F1 = score([output_text], [reference_text], lang="en")
        metrics["bert_score"] = float(F1[0])
    except ImportError:
        pass

    # Hallucination detection (simple keyword‑based)
    hallucination_phrases = ["I am not sure", "might be", "possibly", "I think"]
    hallucination_score = sum(
        1 for p in hallucination_phrases if p.lower() in output_text.lower()
    )
    metrics["hallucination_risk"] = hallucination_score

    # Toxicity
    try:
        from transformers import pipeline

        toxicity = pipeline("text-classification", model="unitary/toxic-bert")
        result = toxicity(output_text)[0]
        metrics["toxicity"] = result["score"] if result["label"] == "toxic" else 0.0
    except:
        metrics["toxicity"] = 0.0

    return metrics


def run_monitors_for_endpoint(endpoint):
    """Run all active monitors for an endpoint."""
    recent_logs = endpoint.logs.order_by("-created_at")[:10]
    for monitor in endpoint.monitors.filter(is_active=True):
        for log in recent_logs:
            metrics = evaluate_llm_output(log.output_data)
            value = metrics.get(monitor.metric, 0)
            # Check thresholds
            if monitor.threshold_critical and value >= monitor.threshold_critical:
                LLMMonitorAlert.objects.create(
                    monitor=monitor,
                    value=value,
                    severity="critical",
                    message=f"{monitor.metric} critical: {value} >= {monitor.threshold_critical}",
                )
            elif monitor.threshold_warning and value >= monitor.threshold_warning:
                LLMMonitorAlert.objects.create(
                    monitor=monitor,
                    value=value,
                    severity="warning",
                    message=f"{monitor.metric} warning: {value} >= {monitor.threshold_warning}",
                )
