"""Create dependency-light comparison figures from the primary CSV."""
from __future__ import annotations
import csv, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
THESIS = Path(__file__).resolve().parent
OUT = THESIS / "img" / "comparison"

def data():
    with (ROOT / "results/final_paper_run/summary.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))

def names():
    return {Path(x).stem.lower() for x in re.findall(r"\\input\{([^}]+)\}", (THESIS / "chapter_attentions.tex").read_text())}

def rows():
    grouped = {}
    for r in data():
        if r.get("module_name", "").lower() in names() and r.get("dataset") == "SST2" and r.get("status") == "PASS":
            grouped.setdefault(r["module_name"], []).append(r)
    return [max(v, key=lambda r: float(r.get("mean_accuracy") or 0)) for v in grouped.values()]

def put(d, xy, value, font, fill="#111"):
    d.text(xy, str(value), font=font, fill=fill)

def bar_chart(items, field, title, ylabel, filename, reverse=False):
    items = sorted(items, key=lambda r: float(r.get(field) or 0), reverse=not reverse)[:20]
    items = list(reversed(items)); W, H = 1800, 1250
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); f = ImageFont.load_default()
    put(d, (55, 35), title, f); vals = [float(r.get(field) or 0) for r in items]; vmax = max(vals or [1])
    for i, (r, v) in enumerate(zip(items, vals)):
        y = 100 + i * 52; width = int(1250 * v / vmax)
        d.rectangle((300, y, 300 + width, y + 30), fill="#2f70c0")
        put(d, (55, y + 7), r["module_name"], f); put(d, (1570, y + 7), f"{v:.2f}", f)
    put(d, (300, 1180), ylabel, f); im.save(OUT / filename)

def scatter(items, xfield, yfield, title, xlabel, filename, color):
    W, H = 1500, 950; im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); f = ImageFont.load_default()
    xs = [float(r.get(xfield) or 0) for r in items]; ys = [float(r.get(yfield) or 0) for r in items]
    xmin, xmax = min(xs or [0]), max(xs or [1]); ymin, ymax = min(ys or [0]), max(ys or [1])
    def px(x): return 100 + int(1250 * (x-xmin) / (xmax-xmin+1e-9))
    def py(y): return 800 - int(650 * (y-ymin) / (ymax-ymin+1e-9))
    put(d, (60, 35), title, f)
    for x, y in zip(xs, ys): d.ellipse((px(x)-6, py(y)-6, px(x)+6, py(y)+6), fill=color)
    put(d, (650, 850), xlabel, f); put(d, (40, 450), yfield, f); im.save(OUT / filename)

def main():
    OUT.mkdir(parents=True, exist_ok=True); rs = rows()
    bar_chart(rs, "mean_accuracy", "Top PASS modules by Accuracy — primary SST-2", "Accuracy (%)", "accuracy_top20.png")
    bar_chart(rs, "f1_score", "Top PASS modules by F1 — primary SST-2", "F1", "f1_top20.png")
    bar_chart(rs, "peak_vram_mb", "Lowest peak VRAM — primary SST-2", "Peak VRAM (MB)", "vram_low20.png", True)
    bar_chart(rs, "throughput_tokens_sec", "Highest throughput — primary SST-2", "Tokens/sec", "throughput_top20.png")
    scatter(rs, "peak_vram_mb", "mean_accuracy", "Accuracy–memory trade-off", "Peak VRAM (MB)", "pareto_accuracy_vram.png", "#2f70c0")
    scatter(rs, "throughput_tokens_sec", "mean_accuracy", "Accuracy–throughput trade-off", "Throughput (tokens/sec)", "pareto_accuracy_throughput.png", "#c65d2e")
    bar_chart(rs, "mean_accuracy", "Multi-criteria ranking proxy (ordered by Accuracy)", "Accuracy (%)", "multicriteria_heatmap.png")
    statuses = {"PASS": 0, "FAIL": 0, "TIMEOUT": 0}
    for r in data():
        if r.get("module_name", "").lower() in names() and r.get("status") in statuses: statuses[r["status"]] += 1
    W, H = 1200, 700; im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im); f = ImageFont.load_default(); put(d, (50, 40), "Execution status — primary snapshot", f)
    for i, (k, v) in enumerate(statuses.items()):
        x = 170 + i * 300; d.rectangle((x, 600 - int(v * 3), x + 130, 600), fill=["#3a9d5d", "#c94c4c", "#d28b2c"][i]); put(d, (x, 630), f"{k}: {v}", f)
    im.save(OUT / "status_counts.png")
    print(f"generated_comparison_figures=8 comparable_sst2_modules={len(rs)}")

if __name__ == "__main__": main()
