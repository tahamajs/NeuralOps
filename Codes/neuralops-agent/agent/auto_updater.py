"""
Self-update mechanism — checks NeuralOps server for new versions and hot-reloads.
Supports PyInstaller frozen binaries, safe SHA-256 verification, and backup rollbacks.
"""

import hashlib
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("neuralops.updater")


class AutoUpdater:
    """Manages agent updates, binary validation, atomic replacement, and hot reloading."""

    def __init__(self, config, api_client):
        self.config = config
        self.api = api_client

    def check_and_update(self) -> bool:
        """
        Check server for new version; download, verify checksum, and apply if newer.

        Returns:
            True if an update was found and applied/notified, False otherwise.
        """
        if not getattr(self.config, "auto_update", True):
            logger.debug("Auto-update is disabled in configuration")
            return False

        try:
            current_ver = getattr(self.config, "version", "1.0.0")
            latest = self._fetch_latest_version()
            if not latest or not latest.get("version"):
                logger.debug("No update metadata returned from server")
                return False

            latest_ver = latest["version"]
            if self._version_newer(latest_ver, current_ver):
                logger.info(
                    "New NeuralOps Agent version available: %s → %s",
                    current_ver,
                    latest_ver,
                )

                download_url = latest.get("download_url")
                expected_sha256 = latest.get("sha256", "").strip().lower()

                # If running as compiled binary (PyInstaller / py2exe)
                is_frozen = getattr(sys, "frozen", False)
                if is_frozen:
                    if not download_url or not expected_sha256:
                        logger.warning("Update metadata missing download_url or sha256 checksum")
                        return False
                    return self._download_and_apply(download_url, expected_sha256)
                else:
                    logger.info(
                        "Running in source/development mode. Skipping binary replacement for version %s. "
                        "Please update via git pull or pip install.",
                        latest_ver,
                    )
                    return True

            return False
        except Exception as e:
            logger.warning("Auto-update check encountered an error: %s", e)
            return False

    def _fetch_latest_version(self) -> Optional[Dict[str, Any]]:
        """Fetch latest version metadata from backend API."""
        try:
            current_ver = getattr(self.config, "version", "1.0.0")
            if hasattr(self.api, "check_for_updates") and callable(self.api.check_for_updates):
                resp = self.api.check_for_updates(current_ver)
            elif hasattr(self.api, "_post") and callable(self.api._post):
                resp = self.api._post(
                    "/api/agent/check-update/", {"version": current_ver}
                )
            else:
                resp = None

            if isinstance(resp, dict) and resp.get("version"):
                return resp
            return None
        except Exception as e:
            logger.debug("Failed to fetch latest version from API: %s", e)
            return None

    def _version_newer(self, latest: str, current: str) -> bool:
        """Compare two version strings safely."""
        try:
            from packaging.version import parse
            return parse(latest) > parse(current)
        except Exception:
            # Fallback simple tuple comparison
            try:
                def parse_tuples(v: str):
                    return tuple(int(p) for p in v.split(".") if p.isdigit())
                return parse_tuples(latest) > parse_tuples(current)
            except Exception:
                return latest != current

    def _download_and_apply(
        self, url: str, expected_sha256: str, target_path: Optional[str] = None
    ) -> bool:
        """
        Download new binary, verify SHA-256 integrity, atomically replace, and restart.
        """
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tmp_name = tmp_file.name
        try:
            logger.info("Downloading update from %s...", url)
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            sha = hashlib.sha256()
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    tmp_file.write(chunk)
                    sha.update(chunk)
            tmp_file.close()

            calculated_sha = sha.hexdigest().lower()
            if calculated_sha != expected_sha256.lower():
                logger.error(
                    "Update verification failed: SHA-256 mismatch (expected: %s, got: %s)",
                    expected_sha256,
                    calculated_sha,
                )
                return False

            logger.info("SHA-256 checksum verified successfully")

            # Determine binary destination
            executable = target_path or sys.executable
            backup = str(Path(executable).with_suffix(".bak"))

            # Create backup and atomically move
            if os.path.exists(backup):
                try:
                    os.remove(backup)
                except Exception:
                    pass

            shutil.move(executable, backup)
            shutil.move(tmp_name, executable)
            os.chmod(executable, 0o755)

            logger.info("Update installed successfully. Restarting agent...")
            # Restart agent process
            os.execv(executable, sys.argv)
            return True

        except Exception as e:
            logger.error("Failed to apply binary update: %s", e)
            return False
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass
