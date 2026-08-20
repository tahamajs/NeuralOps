import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove Access-Control-Allow-Credentials
content = content.replace('self.send_header("Access-Control-Allow-Credentials", "true")\n', '')
content = content.replace('self.send_header("Access-Control-Allow-Credentials", "true")', '')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
