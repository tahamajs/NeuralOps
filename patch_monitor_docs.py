import re

with open("Codes/framework/live_monitor.py", "r") as f:
    content = f.read()

# Replace BASE_DIR with WORKDIR
content = content.replace("BASE_DIR.parent", "WORKDIR.parent")

with open("Codes/framework/live_monitor.py", "w") as f:
    f.write(content)
