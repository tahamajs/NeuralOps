import re

filepath = "D:/SVD_Project/auto_git_push.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

watchdog_code = """
        import psutil
        import time
        # Watchdog: Kill any run_quality_sweep.py that has been running for > 20 minutes (1200 seconds)
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = p.info['cmdline']
                if cmdline and 'python' in p.info['name'].lower() and any('run_quality_sweep.py' in arg for arg in cmdline):
                    uptime = time.time() - p.info['create_time']
                    if uptime > 1200:
                        print(f"Watchdog: Killing stuck process {p.info['pid']} (uptime {uptime}s)")
                        p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
"""

if "psutil" not in content:
    content = content.replace("try:\n", "try:\n" + watchdog_code)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched auto_git_push.py with watchdog.")
