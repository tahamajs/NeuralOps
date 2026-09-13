#!/usr/bin/env python3
"""
build_standalone.py
===================
Compiles index.html + style.css + app.js + spatial-data.json into a single
completely self-contained, zero-dependency standalone.html distribution.
Can be opened directly via double-click without a web server.
"""

import os
import re
import json

def build_standalone():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    css_path = os.path.join(base_dir, "style.css")
    js_path = os.path.join(base_dir, "app.js")
    data_path = os.path.join(base_dir, "spatial-data.json")
    standalone_path = os.path.join(base_dir, "standalone.html")

    print("[Building standalone.html]")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()

    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    with open(data_path, "r", encoding="utf-8") as f:
        data_json = f.read()

    catalog_path = os.path.join(base_dir, "modules-catalog.json")
    catalog_json = "{}"
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog_json = f.read()

    components_path = os.path.join(base_dir, "modules_components.json")
    components_json = "{}"
    if os.path.exists(components_path):
        with open(components_path, "r", encoding="utf-8") as f:
            components_json = f.read()

    # 1. Inline CSS
    html = html.replace(
        '<link rel="stylesheet" href="style.css">',
        f'<style>\n{css}\n</style>'
    )

    # 2. Embed Spatial Data, Module Catalog, Components, and Inline JS
    embedded_data_script = f"""
    <script>
        window.__EMBEDDED_SPATIAL_DATA__ = {data_json};
        window.__EMBEDDED_MODULES_CATALOG__ = {catalog_json};
        window.__EMBEDDED_MODULES_COMPONENTS__ = {components_json};
    </script>
    <script type="module">
{js}
    </script>
    """

    html = html.replace(
        '<script type="module" src="app.js"></script>',
        embedded_data_script
    )

    # Update page title to indicate standalone edition
    html = html.replace(
        '<title>3D Neural Architecture Visualizer · Multi-Model Atlas</title>',
        '<title>3D Neural Architecture Visualizer · Standalone Edition</title>'
    )

    with open(standalone_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(standalone_path) / 1024
    print(f"[OK] Successfully built {standalone_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build_standalone()
