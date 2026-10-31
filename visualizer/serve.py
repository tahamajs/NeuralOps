#!/usr/bin/env python3
"""
serve.py
========
One-click local server and browser launcher for the 3D Neural Architecture Visualizer.
Features:
- Automatically finds an open port if 8000 is occupied.
- Auto-opens browser to the visualizer.
- Zero configuration required.
- Clean Ctrl+C shutdown without stack traces or socket leaks.
"""

import os
import sys
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

DEFAULT_PORT = 8000

class VisualizerHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress verbose favicon 404 noise in terminal
        if 'favicon.ico' in (args[0] if args else ''):
            return
        super().log_message(format, *args)


def find_available_port(starting_port=8000, max_attempts=50):
    for port in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return starting_port


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = find_available_port(DEFAULT_PORT)

    HTTPServer.allow_reuse_address = True
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, VisualizerHandler)
    except OSError as e:
        print(f"Error binding to port {port}: {e}")
        sys.exit(1)

    url = f"http://localhost:{port}/index.html"
    print("\n" + "=" * 60)
    print("  🚀 3D Neural Architecture Visualizer Server Running")
    if port != DEFAULT_PORT:
        print(f"  ℹ️  Port {DEFAULT_PORT} was in use, automatically switched to {port}!")
    print(f"  👉 URL: {url}")
    print("  Press Ctrl+C to stop the server anytime.")
    print("=" * 60 + "\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server stopped cleanly]")
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
