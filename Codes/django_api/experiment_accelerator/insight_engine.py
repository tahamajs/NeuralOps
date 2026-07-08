"""
AI Insight Engine — analyzes experiment history and generates insights.
Uses simple statistical methods (no external API needed).
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def analyze_user_experiments(user):
    """
    Analyze a user's experiments and return a dict of insights.
    """
    from experiment.models import TrainingRun

    runs = TrainingRun.objects.filter(user=user, status="completed").order_by(
        "-created_at"
    )

    if not runs.exists():
        return []

    insights = []

    # 1. Trend analysis — are we improving over time?
    trend = _detect_improvement_trend(runs)
    if trend:
        insights.append(trend)

    # 2. Winning pattern — what makes the best runs special?
    winner = _detect_winning_pattern(runs)
    if winner:
        insights.append(winner)

    # 3. Optimization suggestion
    optimization = _suggest_optimization(runs)
    if optimization:
        insights.append(optimization)

    # 4. Anomaly detection
    anomaly = _detect_anomaly(runs)
    if anomaly:
        insights.append(anomaly)

    return insights


def _detect_improvement_trend(runs):
    """Check if accuracy is improving over recent runs."""
    recent = [
        r for r in runs if r.final_metrics and r.final_metrics.get("eval_accuracy")
    ]
    if len(recent) < 3:
        return None

    recent.sort(key=lambda r: r.created_at)
    accuracies = [r.final_metrics["eval_accuracy"] for r in recent]

    # Simple linear trend: compare first half vs second half
    mid = len(accuracies) // 2
    first_half_avg = sum(accuracies[:mid]) / mid
    second_half_avg = sum(accuracies[mid:]) / (len(accuracies) - mid)

    improvement = (second_half_avg - first_half_avg) / max(first_half_avg, 0.001)

    if improvement > 0.05:
        return {
            "insight_type": "trend",
            "title": "📈 Accuracy is improving!",
            "description": f"Your recent experiments show a {improvement*100:.1f}% improvement in accuracy compared to earlier runs. Keep going with the current approach.",
            "confidence": min(0.9, 0.5 + improvement),
            "metadata": {
                "first_half_avg": round(first_half_avg, 4),
                "second_half_avg": round(second_half_avg, 4),
                "improvement_pct": round(improvement * 100, 1),
            },
        }
    elif improvement < -0.03:
        return {
            "insight_type": "trend",
            "title": "📉 Accuracy is declining",
            "description": f"Recent experiments are performing worse than earlier ones ({abs(improvement)*100:.1f}% drop). Consider reverting to earlier configurations or trying a different approach.",
            "confidence": min(0.8, 0.5 + abs(improvement)),
            "metadata": {
                "first_half_avg": round(first_half_avg, 4),
                "second_half_avg": round(second_half_avg, 4),
                "decline_pct": round(abs(improvement) * 100, 1),
            },
        }
    return None


def _detect_winning_pattern(runs):
    """Find what the best performing runs have in common."""
    completed = [
        r for r in runs if r.final_metrics and r.final_metrics.get("eval_accuracy")
    ]
    if len(completed) < 3:
        return None

    # Sort by accuracy
    completed.sort(key=lambda r: r.final_metrics["eval_accuracy"], reverse=True)
    top_runs = completed[: max(1, len(completed) // 3)]

    # Analyze common config patterns in top runs
    configs = [r.config or {} for r in top_runs]

    patterns = {}
    for key in ["attention_type", "method", "batch_size", "learning_rate", "epochs"]:
        values = [c.get(key) for c in configs if c.get(key) is not None]
        if values:
            most_common = Counter(values).most_common(1)[0]
            if most_common[1] >= len(top_runs) * 0.5:
                patterns[key] = most_common[0]

    if patterns:
        pattern_desc = ", ".join(f"{k}={v}" for k, v in patterns.items())
        return {
            "insight_type": "winner",
            "title": "🏆 Winning pattern detected",
            "description": f"Your best experiments share these traits: {pattern_desc}. Consider using these settings as a baseline.",
            "confidence": 0.75,
            "metadata": {"patterns": patterns},
        }
    return None


def _suggest_optimization(runs):
    """Suggest an optimization based on what hasn't been tried."""
    completed = [r for r in runs if r.final_metrics and r.config]
    if len(completed) < 3:
        return None

    # What attention types have been tried?
    tried_attention = set(
        r.config.get("attention_type")
        for r in completed
        if r.config.get("attention_type")
    )
    tried_methods = set(
        r.config.get("method") for r in completed if r.config.get("method")
    )

    # Suggest untried attention types
    all_attention = {
        "full",
        "galerkin",
        "linformer",
        "spectraformer",
        "padre",
        "aurora",
        "nova",
    }
    untried = all_attention - tried_attention
    if untried:
        suggestion = list(untried)[0]
        return {
            "insight_type": "optimization",
            "title": f"💡 Try a new attention type: {suggestion}",
            "description": f"You haven't experimented with the '{suggestion}' attention mechanism yet. Researchers on similar tasks have seen improvements with it.",
            "confidence": 0.6,
            "metadata": {"suggestion": suggestion, "category": "attention_type"},
        }
    return None


def _detect_anomaly(runs):
    """Detect runs that are outliers (much worse or better than average)."""
    completed = [
        r for r in runs if r.final_metrics and r.final_metrics.get("eval_accuracy")
    ]
    if len(completed) < 5:
        return None

    accuracies = [r.final_metrics["eval_accuracy"] for r in completed]
    avg = sum(accuracies) / len(accuracies)
    std = (sum((a - avg) ** 2 for a in accuracies) / len(accuracies)) ** 0.5

    # Find outliers (> 2 std away)
    for r in completed:
        acc = r.final_metrics["eval_accuracy"]
        if abs(acc - avg) > 2 * std:
            direction = "higher" if acc > avg else "lower"
            return {
                "insight_type": "anomaly",
                "title": "🔍 Unusual result detected",
                "description": f"One of your experiments achieved an accuracy of {acc:.4f}, which is significantly {direction} than your average of {avg:.4f}. Investigate what made this run different.",
                "confidence": 0.7,
                "metadata": {
                    "anomaly_run_id": str(r.id),
                    "anomaly_value": acc,
                    "average": round(avg, 4),
                    "std": round(std, 4),
                },
            }
    return None
