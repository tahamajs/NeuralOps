import json
from pathlib import Path

def main():
    count = 0
    pnc_list = []
    for f in Path("results").rglob("*.json"):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                d = json.load(fp)
            if isinstance(d, dict) and d.get("performance", {}).get("status") == "PASS_NO_COMPRESSION":
                count += 1
                perf = d.get("performance", {})
                eff = d.get("efficiency", {})
                pnc_list.append({
                    "file": str(f).replace("\\", "/"),
                    "name": f.name,
                    "model": d.get("model", ""),
                    "ppl": perf.get("perplexity", "N/A"),
                    "acc": perf.get("accuracy_pct", "N/A"),
                    "compressed_layers": eff.get("num_compressed_layers", 0),
                    "fallback_layers": eff.get("num_fallback_layers", 0),
                })
        except Exception:
            pass

    for item in pnc_list:
        print(f"{item['name']} -> ppl={item['ppl']} acc={item['acc']} compressed={item['compressed_layers']} fallback={item['fallback_layers']}")
    print(f"Total PASS_NO_COMPRESSION: {count}")

if __name__ == "__main__":
    main()
