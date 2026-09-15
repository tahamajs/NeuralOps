import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_variables(path: str = "variables.json"):
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def render_section(section_name: str, values: dict) -> str:
    template = (ROOT / f"{section_name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def main():
    variables = load_variables()
    for section in [
        "00_meta",
        "01_intro",
        "02_related",
        "03_background",
        "04_method",
        "05_architecture",
        "06_setup",
        "07_results",
        "08_analysis",
        "09_discussion",
        "10_limitations",
        "11_conclusion",
        "12_reproducibility",
        "13_references",
        "A_math",
        "B_implementation",
        "C_data",
        "D_negative_results",
        "E_ethics",
    ]:
        render_section(section, variables)


if __name__ == "__main__":
    main()
