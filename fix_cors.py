import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add more permissive CORS headers
content = content.replace(
    'self.send_header("Access-Control-Allow-Origin", "*")',
    'self.send_header("Access-Control-Allow-Origin", "*")\n            self.send_header("Access-Control-Allow-Credentials", "true")'
)

# And replace the exact fetch URL if the user is using file://
html_fix = """
            const res = await fetch('http://192.168.1.100:8080/api/stats');
"""
content = re.sub(r"const res = await fetch\('/api/stats'\);", html_fix, content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
