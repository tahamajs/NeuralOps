import csv
from pathlib import Path

def main():
    base = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")
    print("MODEL|MODULE|STATUS|ACCURACY|RECON_ERROR|SPEED_TOK_S|COMPRESSION|VRAM_MB")
    print("-" * 75)
    for model in ["distilbert", "roberta"]:
        p = base / "results" / f"batch3_{model}"
        if not p.exists():
            continue
        for mod_dir in sorted(p.iterdir()):
            csv_path = mod_dir / "summary.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, "r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            m_name = row.get("module_name", mod_dir.name)
                            status = row.get("status", "N/A")
                            acc = row.get("accuracy_pct", "N/A")
                            recon = row.get("avg_reconstruction_error", "N/A")
                            speed = row.get("tokens_per_sec", "N/A")
                            comp = row.get("compression_rate", "N/A")
                            vram = row.get("peak_vram_mb", "N/A")
                            print(f"{model.upper()}|{m_name}|{status}|{acc}|{recon}|{speed}|{comp}|{vram}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()
