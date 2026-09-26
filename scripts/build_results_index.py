import json

with open("results/RESULTS_MASTER_INDEX.json") as f:
    idx = json.load(f)

cats = {}
for name, data in idx.items():
    c = data["category"]
    cats.setdefault(c, []).append((name, data))

lines = []
lines.append("# 🗂 Results Master Directory & Quick-Access Catalog\n")
lines.append("This catalog provides an instant directory index and quick-access map for all **268 benchmark suites** stored under `results/`.\n")
lines.append("You can locate any specific experiment, summary CSV, or JSON artifact instantly using the categorized tables below.\n")
lines.append("---\n")

for cat_name, items in sorted(cats.items()):
    lines.append(f"## 📁 {cat_name} ({len(items)} suites)\n")
    lines.append("| Directory Name | Relative Path | Subdirs | CSVs | JSONs | Primary Summary File |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
    for name, data in sorted(items, key=lambda x: x[0]):
        sum_file = data["summary_files"][0] if data["summary_files"] else "-"
        n_sub = data["num_subdirs"]
        n_csv = data["num_csvs"]
        n_json = data["num_jsons"]
        lines.append(f"| `{name}` | [`results/{name}`](./{name}) | {n_sub} | {n_csv} | {n_json} | `{sum_file}` |")
    lines.append("\n---\n")

with open("results/INDEX.md", "w") as f:
    f.write("\n".join(lines))

print("Regenerated results/INDEX.md cleanly.")
