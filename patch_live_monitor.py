import re

with open("Codes/framework/live_monitor.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the index.html routing
old_routing = """        if req_path in ("/", "/index.html"):
            index_path = WEB_DIR / "index.html"
            if not index_path.exists():
                index_path = PROJECT_DIR / "Docs" / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
                return"""

new_routing = """        if req_path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return"""

content = content.replace(old_routing, new_routing)

with open("Codes/framework/live_monitor.py", "w", encoding="utf-8") as f:
    f.write(content)
