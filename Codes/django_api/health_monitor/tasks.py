"""Celery tasks for periodic drift checks."""
import numpy as np
from celery import shared_task
from .models import DriftMonitor, DriftCheck
from .drift_detector import DriftDetector
from .alerting import AlertingEngine


@shared_task
def run_scheduled_drift_checks():
    """
    Run all active drift monitors that are due for a check.
    """
    from django.utils import timezone
    now = timezone.now()
    monitors = DriftMonitor.objects.filter(is_active=True)

    count = 0
    for monitor in monitors:
        if monitor.last_checked_at is None or \
           (now - monitor.last_checked_at).total_seconds() >= monitor.check_interval_minutes * 60:
            # In production, you'd fetch reference and current data from the feature store
            # Here we use synthetic data for demonstration
            reference = np.random.normal(0, 1, 1000)
            current = np.random.normal(0.1, 1.1, 1000)  # slight drift

            detector = DriftDetector()
            metric_value = detector.detect(monitor.metric, reference, current)

            if metric_value >= monitor.critical_threshold:
                status = 'critical'
            elif metric_value >= monitor.warning_threshold:
                status = 'warning'
            else:
                status = 'healthy'

            check = DriftCheck.objects.create(
                monitor=monitor,
                metric_value=round(metric_value, 4),
                threshold_value=monitor.warning_threshold,
                status=status,
            )

            monitor.last_checked_at = now
            monitor.last_status = status
            monitor.save(update_fields=['last_checked_at', 'last_status'])

            if status in ('warning', 'critical'):
                engine = AlertingEngine()
                engine.trigger_alert(monitor, check)

            count += 1

    return count