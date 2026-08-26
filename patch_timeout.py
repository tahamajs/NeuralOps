from pathlib import Path

file_path = Path('Codes/framework/live_monitor.py')
content = file_path.read_text()

content = content.replace("res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)", "res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)")
file_path.write_text(content)
print("timeout patched")
