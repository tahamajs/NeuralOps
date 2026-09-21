"""
Watchdog for auto_pipeline.py.
Monitors the daemon process every 5 minutes and automatically restarts it if it crashes.
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "watchdog.log")
PIPELINE_SCRIPT = os.path.join(ROOT_DIR, "Codes", "framework", "auto_pipeline.py")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("watchdog")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)


def is_auto_pipeline_running() -> bool:
    try:
        if sys.platform == "win32":
            # Windows PowerShell / tasklist check
            out = subprocess.check_output(
                ["powershell", "-Command", "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
                encoding="utf-8",
                errors="ignore",
                timeout=10,
            )
            return "auto_pipeline" in out
        else:
            # POSIX check
            out = subprocess.check_output(["pgrep", "-f", "auto_pipeline.py"], encoding="utf-8", errors="ignore", timeout=5)
            return bool(out.strip())
    except Exception as e:
        logger.debug(f"Process check failed: {e}")
        return False


def restart_pipeline() -> None:
    logger.warning("auto_pipeline.py is NOT running! Restarting daemon now...")
    python_bin = sys.executable
    if sys.platform == "win32":
        cmd = f'start /B "{python_bin}" -u "{PIPELINE_SCRIPT}" > "{os.path.join(ROOT_DIR, "logs", "auto_pipeline.log")}" 2>&1'
        subprocess.Popen(cmd, shell=True, cwd=ROOT_DIR)
    else:
        log_f = open(os.path.join(ROOT_DIR, "logs", "auto_pipeline.log"), "a", encoding="utf-8")
        subprocess.Popen([python_bin, "-u", PIPELINE_SCRIPT], stdout=log_f, stderr=log_f, cwd=ROOT_DIR, start_new_session=True)
    logger.info("Restart command executed.")


def main() -> None:
    logger.info("Watchdog started for auto_pipeline.py (checking every 300s).")
    while True:
        try:
            alive = is_auto_pipeline_running()
            if alive:
                logger.info("Watchdog health check: auto_pipeline.py is ALIVE.")
            else:
                restart_pipeline()
        except Exception as e:
            logger.error(f"Watchdog exception: {e}", exc_info=True)
        time.sleep(300)


if __name__ == "__main__":
    main()
