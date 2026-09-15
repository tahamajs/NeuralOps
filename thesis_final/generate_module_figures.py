"""Generate auditable per-module figures from source features and the primary CSV."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
THESIS = Path(__file__).resolve().parent
OUT = THESIS / "img" / "modules"
CSV = ROOT / "results" / "final_paper_run" / "summary.csv"
SOURCE_DIRS = [ROOT / "Codes" / "framework" / "attentions", ROOT / "Codes" / "framework" / "compression_engine"]


def source_for(stem: str) -> Path | None:
    aliases = {"tt": "token_compressed_attention", "eat_attention": "eat", "dobi_attention": "dobi", "padre": "padre_attention", "compressed_mha": "compressed_mha_attention"}
    lookup = aliases.get(stem.lower(), stem).lower()
    files = [p for d in SOURCE_DIRS for p in d.glob("*.py")]
    exact = [p for p in files if p.stem.lower() == lookup]
    return sorted(exact)[0] if exact else None


def features(text: str) -> list[str]:
    checks = [("SVD", r"linalg\.svd|torch\.svd|singular"), ("LoRA", r"lora"),
              ("Top-K/sparse", r"topk|top_k|sparse|block_size"), ("RoPE", r"rotary|rope"),
              ("GQA", r"repeat_kv|num_kv_heads|grouped"), ("state/scan", r"scan|recurrence|state|decay|delta"),
              ("Softmax", r"softmax")]
    return [label for label, pat in checks if re.search(pat, text, re.I)]


def rows_by_module() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "PASS":
                out.setdefault(row.get("module_name", "").lower(), []).append(row)
    return out


def draw(stem: str, source: Path | None, rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8", errors="ignore") if source else ""
    feats = features(text)
    canvas = Image.new("RGB", (1800, 900), "white")
    ax = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bold = ImageFont.load_default()
    labels = ["Input X", "Projection", "Module core", "Output"]
    if "state/scan" in feats:
        labels[2] = "State / scan"
    elif "SVD" in feats:
        labels[2] = "SVD / low-rank"
    elif "Top-K/sparse" in feats:
        labels[2] = "Sparse / Top-K"
    elif "LoRA" in feats:
        labels[2] = "LoRA adapter"
    xs = [180, 620, 1120, 1600]
    for x, label in zip(xs, labels):
        box = (x - 135, 390, x + 135, 510)
        ax.rounded_rectangle(box, radius=24, fill="#eaf2ff", outline="#2864b5", width=4)
        tw = ax.textbbox((0, 0), label, font=bold)[2]
        ax.text(((box[0] + box[2] - tw) / 2, 440), label, fill="#111", font=bold)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.line((a + 140, 450, b - 150, 450), fill="#444", width=5)
        ax.polygon([(b - 150, 450), (b - 170, 438), (b - 170, 462)], fill="#444")
    ax.text((90, 90), f"Code-derived tensor path: {stem}", fill="#111", font=bold)
    ax.text((90, 690), "Detected code operations: " + (", ".join(feats) if feats else "no special pattern detected"), fill="#333", font=font)
    canvas.save(OUT / f"{stem}_path.png")

    canvas = Image.new("RGB", (1300, 800), "white")
    ax = ImageDraw.Draw(canvas)
    ax.text((80, 60), f"Primary PASS results: {stem}", fill="#111", font=bold)
    if rows:
        vals = [float(r.get("mean_accuracy") or 0) for r in rows]
        width = max(80, 950 // max(1, len(vals)))
        for i, (v, row) in enumerate(zip(vals, rows)):
            x = 150 + i * (width + 35)
            top = 680 - int(5.2 * v)
            ax.rectangle((x, top, x + width, 680), fill="#2f70c0")
            ax.text((x, top - 28), f"{v:.2f}", fill="#111", font=font)
            ax.text((x, 710), f"{row.get('model_family','')} {row.get('dataset','')}", fill="#333", font=font)
        ax.line((100, 680, 1150, 680), fill="#333", width=3)
        ax.text((80, 220), "Accuracy (%)", fill="#333", font=font)
    else:
        ax.text((400, 400), "No PASS row in primary CSV - Pending Verification", fill="#111", font=bold)
    canvas.save(OUT / f"{stem}_results.png")


def main() -> None:
    rows = rows_by_module()
    chapter = (THESIS / "chapter_attentions.tex").read_text(encoding="utf-8")
    stems = [Path(x).stem for x in re.findall(r"\\input\{([^}]+)\}", chapter)]
    for stem in stems:
        draw(stem, source_for(stem), rows.get(stem.lower(), []))
    print(f"generated_figures={2 * len(stems)} modules={len(stems)}")


if __name__ == "__main__":
    main()
