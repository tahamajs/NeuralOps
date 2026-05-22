"""Alert & notification engine – sends alerts via multiple channels."""
import logging
import requests
from django.conf import settings
from django.core.mail import send_mail
from .models import Alert, AlertChannel, DriftCheck, DriftMonitor

logger = logging.getLogger(__name__)


class AlertingEngine:
    """Handles alert creation and delivery."""

    def trigger_alert(self, monitor: DriftMonitor, check: DriftCheck):
        """Create and send an alert when a threshold is breached."""
        severity = 'critical' if check.status == 'critical' else 'warning'
        title = f"[{severity.upper()}] {monitor.name} – Drift Detected"
        message = (
            f"Monitor: {monitor.name}\n"
            f"Type: {monitor.drift_type}\n"
            f"Metric: {monitor.metric}\n"
            f"Value: {check.metric_value:.4f}\n"
            f"Threshold: {check.threshold_value:.4f}\n"
            f"Time: {check.checked_at}\n"
        )

        alert = Alert.objects.create(
            monitor=monitor,
            drift_check=check,
            severity=severity,
            channel='in_app',
            title=title,
            message=message,
        )

        # Send to configured alert channels
        channels = AlertChannel.objects.filter(is_active=True)
        for channel in channels:
            self._send_to_channel(channel, title, message, severity, alert)

        return alert

    def _send_to_channel(self, channel: AlertChannel, title: str, message: str, severity: str, alert: Alert):
        """Send an alert to a specific channel."""
        if channel.channel_type == 'email':
            self._send_email(channel.destination, title, message)
        elif channel.channel_type == 'slack':
            self._send_slack(channel.destination, title, message, severity)
        elif channel.channel_type == 'webhook':
            self._send_webhook(channel.destination, title, message, severity)

    def _send_email(self, to_email: str, subject: str, body: str):
        try:
            send_mail(
                subject=subject,
                message=body,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[to_email],
                fail_silently=True,
            )
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack(self, webhook_url: str, title: str, message: str, severity: str):
        color = '#ff0000' if severity == 'critical' else '#ffa500'
        payload = {
            'attachments': [{
                'color': color,
                'title': title,
                'text': message,
            }]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_webhook(self, url: str, title: str, message: str, severity: str):
        payload = {'title': title, 'message': message, 'severity': severity}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")