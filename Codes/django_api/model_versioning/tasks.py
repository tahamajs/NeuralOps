"""
Celery tasks for model versioning app.
"""

import logging

from celery import shared_task

from .models import ModelSecurityScan, ModelUsageLog

logger = logging.getLogger(__name__)


@shared_task
def run_security_scan(scan_id: str):
    """
    Simulate a security scan for a ModelSecurityScan record.
    In production you would call an external scanner (e.g. bandit, safety, trivy).
    """
    try:
        scan = ModelSecurityScan.objects.get(id=scan_id)
    except ModelSecurityScan.DoesNotExist:
        logger.error(f"Security scan {scan_id} not found")
        return

    scan.status = "running"
    scan.save(update_fields=["status"])

    try:
        # ------------------------------------------------------------
        # Replace this block with a real scan implementation
        # ------------------------------------------------------------
        # Example: scanning dependency files in the model artifact
        # import subprocess, json
        # result = subprocess.run(
        #     ['safety', 'check', '--json'],
        #     capture_output=True, text=True
        # )
        # findings = json.loads(result.stdout)

        # Simulated findings
        findings = {
            "vulnerabilities": [],
            "licenses": ["MIT", "Apache-2.0"],
            "secrets_found": False,
        }
        scan.findings = findings
        scan.severity = "low"  # or derive from findings
        scan.passed = True
        scan.status = "completed"
        scan.report_url = ""  # set if you generate a report
    except Exception as e:
        logger.exception(f"Scan {scan_id} failed: {e}")
        scan.status = "failed"
        scan.findings = {"error": str(e)}
        scan.passed = False

    scan.save()


@shared_task
def log_model_usage_async(
    version_id: str,
    user_id: int,
    action: str,
    metadata: dict = None,
    ip_address: str = None,
):
    """Async wrapper to log model usage (useful for high‑traffic endpoints)."""
    from django.contrib.auth import get_user_model

    from .models import ModelVersion

    User = get_user_model()

    try:
        version = ModelVersion.objects.get(id=version_id)
        user = User.objects.get(id=user_id)
    except (ModelVersion.DoesNotExist, User.DoesNotExist):
        logger.warning(f"Could not log usage: version={version_id}, user={user_id}")
        return

    ModelUsageLog.objects.create(
        version=version,
        user=user,
        action=action,
        metadata=metadata or {},
        ip_address=ip_address,
    )
