from pathlib import Path

file_path = Path('Codes/framework/live_monitor.py')
content = file_path.read_text()

content = content.replace("720", "240")
file_path.write_text(content)
print("combos patched")
