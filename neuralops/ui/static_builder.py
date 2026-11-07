#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Static HTML site exporter for offline distribution and HuggingFace Spaces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates

from neuralops.ui.queries.ledger import (
    get_attention_results,
    get_compressor_results,
    get_doctor_status,
    get_recent_runs,
    get_results,
    list_tables,
)
from neuralops.ui.queries.models_notebooks import list_model_zoo, list_notebooks
from neuralops.ui.queries.static_catalog import (
    get_attention_metadata,
    get_compressor_metadata,
    list_attentions,
    list_compressors,
    list_families,
    list_paradigms,
)


class DummyRequest:
    """Mock Starlette Request for offline Jinja2 template rendering."""

    def __init__(self, path: str = "/"):
        self.url = f"http://localhost{path}"
        self.scope = {"type": "http", "path": path}


def build_static_site(output_dir: Path | str) -> dict[str, Any]:
    """Generate a self-contained static HTML site bundle with pre-rendered pages and assets.

    Args:
        output_dir: Target destination directory.

    Returns:
        Summary dictionary containing generated pages count and output path.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    package_root = Path(__file__).resolve().parent
    templates_dir = package_root / "templates"
    static_dir = package_root / "static"

    # Copy static assets (CSS, HTMX, Plot ESM, charts.js)
    out_static = out / "static"
    out_static.mkdir(parents=True, exist_ok=True)
    if static_dir.is_dir():
        for src_file in static_dir.iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                shutil.copy2(src_file, out_static / src_file.name)

    templates = Jinja2Templates(directory=str(templates_dir))
    rendered_files: list[str] = []

    # 1. Dashboard (index.html)
    status = get_doctor_status()
    recent_runs = get_recent_runs(limit=10)
    tpl_dash = templates.get_template("dashboard.html")
    html_dash = tpl_dash.render(
        request=DummyRequest("/"),
        active_tab="dashboard",
        status=status,
        recent_runs=recent_runs,
    )
    (out / "index.html").write_text(html_dash, encoding="utf-8")
    rendered_files.append("index.html")

    # 2. Attentions catalog (attentions.html)
    families = list_families()
    all_attentions = list_attentions()
    tpl_att = templates.get_template("attentions.html")
    html_att = tpl_att.render(
        request=DummyRequest("/attentions"),
        active_tab="attentions",
        attentions=all_attentions,
        families=families,
        selected_family="",
        total_count=len(all_attentions),
    )
    (out / "attentions.html").write_text(html_att, encoding="utf-8")
    rendered_files.append("attentions.html")

    # 2b. Attention details (attentions/<name>.html)
    out_att_dir = out / "attentions"
    out_att_dir.mkdir(parents=True, exist_ok=True)
    tpl_att_detail = templates.get_template("attention_detail.html")
    for att in all_attentions:
        meta = get_attention_metadata(att["name"])
        history = get_attention_results(att["name"])
        html_detail = tpl_att_detail.render(
            request=DummyRequest(f"/attentions/{att['name']}"),
            active_tab="attentions",
            attention=meta,
            history=history,
        )
        (out_att_dir / f"{att['name']}.html").write_text(html_detail, encoding="utf-8")
        rendered_files.append(f"attentions/{att['name']}.html")

    # 3. Compressors catalog (compressors.html)
    paradigms = list_paradigms()
    all_compressors = list_compressors()
    tpl_comp = templates.get_template("compressors.html")
    html_comp = tpl_comp.render(
        request=DummyRequest("/compressors"),
        active_tab="compressors",
        compressors=all_compressors,
        paradigms=paradigms,
        selected_paradigm="",
        total_count=len(all_compressors),
    )
    (out / "compressors.html").write_text(html_comp, encoding="utf-8")
    rendered_files.append("compressors.html")

    # 3b. Compressor details (compressors/<name>.html)
    out_comp_dir = out / "compressors"
    out_comp_dir.mkdir(parents=True, exist_ok=True)
    tpl_comp_detail = templates.get_template("compressor_detail.html")
    for comp in all_compressors:
        comp_meta = get_compressor_metadata(comp["name"])
        comp_history = get_compressor_results(comp["name"])
        html_comp_det = tpl_comp_detail.render(
            request=DummyRequest(f"/compressors/{comp['name']}"),
            active_tab="compressors",
            compressor=comp_meta,
            history=comp_history,
        )
        (out_comp_dir / f"{comp['name']}.html").write_text(html_comp_det, encoding="utf-8")
        rendered_files.append(f"compressors/{comp['name']}.html")

    # 4. Benchmarks explorer (benchmarks.html)
    b_rows, b_total = get_results(limit=100, offset=0)
    tpl_bench = templates.get_template("benchmarks.html")
    html_bench = tpl_bench.render(
        request=DummyRequest("/benchmarks"),
        active_tab="benchmarks",
        rows=b_rows,
        total_count=b_total,
        page=1,
        page_size=100,
        total_pages=max(1, (b_total + 99) // 100),
        families=families,
        filter_module_name="",
        filter_status="",
        filter_family="",
    )
    (out / "benchmarks.html").write_text(html_bench, encoding="utf-8")
    rendered_files.append("benchmarks.html")

    # 5. Model Zoo (models.html)
    models = list_model_zoo()
    tpl_models = templates.get_template("models.html")
    html_models = tpl_models.render(
        request=DummyRequest("/models"),
        active_tab="models",
        models=models,
        total_count=len(models),
    )
    (out / "models.html").write_text(html_models, encoding="utf-8")
    rendered_files.append("models.html")

    # 6. Notebooks (notebooks.html)
    notebooks = list_notebooks()
    tpl_nb = templates.get_template("notebooks.html")
    html_nb = tpl_nb.render(
        request=DummyRequest("/notebooks"),
        active_tab="notebooks",
        notebooks=notebooks,
        total_count=len(notebooks),
        search_query="",
    )
    (out / "notebooks.html").write_text(html_nb, encoding="utf-8")
    rendered_files.append("notebooks.html")

    # 7. Doctor (doctor.html)
    tables = list_tables()
    invariants_passed = (
        status.get("attentions") == 131
        and status.get("compressors") == 72
        and status.get("triton_modules") == 58
        and status.get("cuda_packages") == 141
        and status.get("notebooks") == 94
    )
    tpl_doc = templates.get_template("doctor.html")
    html_doc = tpl_doc.render(
        request=DummyRequest("/doctor"),
        active_tab="doctor",
        status=status,
        tables=tables,
        invariants_passed=invariants_passed,
    )
    (out / "doctor.html").write_text(html_doc, encoding="utf-8")
    rendered_files.append("doctor.html")

    # Write manifest
    manifest = {
        "generator": "neuralops.ui.static_builder",
        "version": status.get("version", "2.0.6"),
        "total_pages": len(rendered_files),
        "files": rendered_files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "output_directory": str(out),
        "total_pages": len(rendered_files),
        "rendered_files": rendered_files,
    }
