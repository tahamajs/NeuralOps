import logging

logger = logging.getLogger(__name__)


def notify(run_id, success, metrics, cfg) -> None:
    """Notify."""
    n = cfg.notification
    if not n:
        return
    if success and "slack" in n.get("on_success", []):
        webhook = n.get("slack_webhook")
        if webhook:
            import requests

            text = f"✅ Run `{run_id}` finished successfully. Acc: {metrics.get('eval_accuracy', 'N/A')}"
            try:
                requests.post(webhook, json={"text": text})
            except Exception:
                logger.exception("Slack notification failed")
    if not success and "email" in n.get("on_failure", []):
        logger.info(
            "Email notification configured – implement with smtplib for production"
        )
