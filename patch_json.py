import re

with open("Codes/framework/live_monitor.py", "r") as f:
    content = f.read()

replacement = """
        elif self.path.endswith(".json"):
            json_path = WEB_DIR / self.path.lstrip('/')
            if json_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json_path.read_bytes())
                return
        
        elif self.path == "/api/stats":
"""

pattern = re.compile(r'\s*elif self\.path == "/api/stats":')
new_content = pattern.sub(replacement, content)

with open("Codes/framework/live_monitor.py", "w") as f:
    f.write(new_content)
