from pathlib import Path
p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")
old = "def _build_stats_response() -> str:\n"
new = "def _build_stats_response() -> str:\n    print('DEBUG ep1 log:', (PROJECT_DIR / 'run_quality_ep1.log').exists(), flush=True)\n    print('DEBUG PROJECT_DIR:', PROJECT_DIR, flush=True)\n"
content = content.replace(old, new)
p.write_text(content, encoding="utf-8")
