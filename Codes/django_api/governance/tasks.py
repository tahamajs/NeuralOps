from celery import shared_task
from .models import ComplianceRule, ComplianceCheck, ComplianceReport, DriftMonitor, DriftAlert, SecurityScan, VulnerabilityDatabase
from .compliance import ComplianceChecker
from .drift import DriftDetector
from .scanner import ModelScanner

@shared_task
def run_compliance_check(content_type: str, object_id: str):
    rules = ComplianceRule.objects.filter(is_active=True)
    results = []
    for rule in rules:
        checker = ComplianceChecker(rule)
        passed, details = checker.check(content_type, object_id)
        ComplianceCheck.objects.create(rule=rule, content_type_id=..., object_id=object_id, passed=passed, details=details)
        results.append(passed)

@shared_task
def generate_weekly_compliance_report():
    """Auto‑generate compliance reports for all users with active contracts."""
    # Implementation: scan all runs, models, contracts, produce Markdown
    pass

@shared_task
def run_drift_check(monitor_id: str):
    monitor = DriftMonitor.objects.get(id=monitor_id)
    detector = DriftDetector(monitor)
    results = detector.check()
    for feature, score in results.items():
        if score > 0.1:  # threshold
            DriftAlert.objects.create(monitor=monitor, feature_name=feature, drift_score=score)

@shared_task
def run_security_scan(scan_id: str):
    scan = SecurityScan.objects.get(id=scan_id)
    scanner = ModelScanner(scan.model_version)
    vulns = scanner.scan()
    scan.vulnerabilities_found = len(vulns)
    scan.details_json = {'vulnerabilities': vulns}
    scan.status = 'completed'
    scan.save()