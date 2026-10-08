"""HTTP Server and request handler implementation for NeuralOps monitor."""

from __future__ import annotations

import http.server
import json
import os
import urllib.parse
from typing import Any

from neuralops.runners.monitor.endpoints import (
    get_doc_content,
    get_loss_history,
    get_module_detail,
    launch_new_run,
    stop_current_run,
)
from neuralops.runners.monitor.rendering import (
    render_csv_export,
    render_favicon,
    render_json_export,
)
from neuralops.runners.monitor.stats import (
    build_aggregated_stats,
    get_module_doc_index,
)
from neuralops.runners.monitor.telemetry import get_live_logs


class ComprehensiveStudioHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler routing telemetry, API, and static file endpoints."""

    project_dir: str = os.getcwd()
    web_dir: str = os.path.join(os.getcwd(), "Docs")
    docs_dir: str = os.path.join(os.getcwd(), "Docs")
    harvested_json: str = os.path.join(os.getcwd(), "results", "harvested_benchmark_data.json")
    is_local_gpu: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self) -> None:
        """Route GET requests to specific API handlers or fall back to static files."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/favicon.ico":
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(render_favicon())
            return

        if path == "/api/stats":
            stats = build_aggregated_stats(
                self.project_dir, self.harvested_json, self.docs_dir, self.is_local_gpu
            )
            self._send_json(stats)
            return

        if path == "/api/logs":
            source = query.get("source", ["active"])[0]
            module_arg = query.get("module", [None])[0]
            lines_arg = int(query.get("lines", ["200"])[0])
            logs_res = get_live_logs(
                self.project_dir, source=source, module=module_arg, lines=lines_arg, is_local_gpu=self.is_local_gpu
            )
            self._send_json(logs_res)
            return

        if path == "/api/loss_history":
            module_arg = query.get("module", [None])[0]
            model_arg = query.get("model", [None])[0]
            loss_res = get_loss_history(
                self.project_dir, module=module_arg, model=model_arg, is_local_gpu=self.is_local_gpu
            )
            self._send_json(loss_res)
            return

        if path == "/api/module_detail":
            mod_name = query.get("module", ["hgrn2_gated_deltanet"])[0]
            detail_res = get_module_detail(self.docs_dir, self.project_dir, mod_name)
            self._send_json(detail_res)
            return

        if path == "/api/run/stop":
            stop_res = stop_current_run()
            self._send_json(stop_res)
            return

        if path == "/api/doc":
            mod_name = query.get("module", ["svd_linear_attention"])[0]
            doc_res = get_doc_content(self.docs_dir, self.project_dir, mod_name)
            self._send_json(doc_res)
            return

        if path == "/api/export/csv":
            stats = build_aggregated_stats(
                self.project_dir, self.harvested_json, self.docs_dir, self.is_local_gpu
            )
            csv_bytes = render_csv_export(stats.get("results", []))
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="neuralops_benchmark_results.csv"'
            )
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(csv_bytes)
            return

        if path == "/api/export/json":
            stats = build_aggregated_stats(
                self.project_dir, self.harvested_json, self.docs_dir, self.is_local_gpu
            )
            json_bytes = render_json_export(stats)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="neuralops_benchmark_results.json"'
            )
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json_bytes)
            return

        if path == "/api/docs_list":
            idx = get_module_doc_index(self.docs_dir)
            clean_list = sorted(list(set(idx.keys())))
            self._send_json({"success": True, "modules": clean_list, "count": len(clean_list)})
            return

        super().do_GET()

    def do_POST(self) -> None:
        """Route POST actions (stop, launch)."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/api/run/stop":
            stop_res = stop_current_run()
            self._send_json(stop_res)
        elif path == "/api/run/launch":
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                params = json.loads(body)
            except Exception:
                params = {}
            launch_res = launch_new_run(params)
            self._send_json(launch_res)
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8080) -> None:
    """Run monitor server forever on target port."""
    import socketserver

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", port), ComprehensiveStudioHandler) as httpd:
        print(f"Full Studio UI & Telemetry API active on http://localhost:{port}")
        httpd.serve_forever()


__all__ = [
    "ComprehensiveStudioHandler",
    "run_server",
]
