"""Self-update mechanism — checks server for new versions and hot-reloads."""

import hashlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger("neuralops.updater")


class AutoUpdater:
    def __init__(self, config, api_client):
        self.config = config
        self.api = api_client

    def check_and_update(self):
        """Check server for new version; download and apply if newer."""
        if not self.config.auto_update:
            return
        try:
            current = self.config.version
            latest = self._fetch_latest_version()
            if not latest:
                return
            if self._version_newer(latest["version"], current):
                logger.info(
                    "New version available: %s → %s", current, latest["version"]
                )
                self._download_and_apply(latest["download_url"], latest["sha256"])
        except Exception as e:
            logger.warning("Update check failed: %s", e)

    def _fetch_latest_version(self):
        resp = self.api._post(
            "/api/agent/check-update/", {"version": self.config.version}
        )
        return resp if resp.get("version") else None

    def _version_newer(self, latest: str, current: str) -> bool:
        from packaging.version import parse

        return parse(latest) > parse(current)

    def _download_and_apply(self, url, sha256):
        """Download new binary, verify hash, replace self."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        try:
            resp = requests.get(url, stream=True)
            sha = hashlib.sha256()
            for chunk in resp.iter_content(8192):
                tmp.write(chunk)
                sha.update(chunk)
            tmp.close()
            if sha.hexdigest() != sha256:
                logger.error("Update hash mismatch — aborting")
                return
            # Replace executable
            current = sys.executable
            backup = str(Path(current).with_suffix(".bak"))
            shutil.move(current, backup)
            shutil.move(tmp.name, current)
            os.chmod(current, 0o755)
            logger.info("Update installed. Restarting...")
            os.execv(current, sys.argv)
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
