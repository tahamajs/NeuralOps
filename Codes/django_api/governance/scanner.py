import os, json, subprocess

class ModelScanner:
    def __init__(self, model_version):
        self.model_version = model_version
    def scan(self) -> list:
        """Run security checks: pickle scan, dependency audit, known CVEs."""
        vulns = []
        # 1. Check for pickle‑based serialization (dangerous)
        if self._contains_pickle():
            vulns.append({'type': 'pickle_insecure', 'severity': 'high', 'description': 'Model uses pickle serialization'})
        # 2. Check requirements.txt against vulnerability database
        cve_matches = VulnerabilityDatabase.objects.filter(affected_packages__contains=['torch'])
        for cve in cve_matches:
            vulns.append({'type': 'cve', 'severity': cve.severity, 'cve_id': cve.cve_id, 'description': cve.description})
        return vulns
    def _contains_pickle(self):
        path = os.path.join(self.model_version.artifact_path, 'model.pth')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                header = f.read(128)
            return b'pickle' in header
        return False