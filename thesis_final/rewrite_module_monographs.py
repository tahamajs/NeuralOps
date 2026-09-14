"""Generate code- and result-aligned Persian module monographs for the thesis.

The generated files deliberately avoid unverified claims and decorative image prompts.
They cite the real source file and the result snapshot used for each metric.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THESIS = Path(__file__).resolve().parent
MODULE_DIR = THESIS / "attentions_latex" / "individual_modules"
SOURCE_DIRS = [ROOT / "Codes" / "framework" / "attentions", ROOT / "Codes" / "framework" / "compression_engine"]
PRIMARY_CSV = ROOT / "results" / "final_paper_run" / "summary.csv"
VERIFIED_JSON = THESIS / "master_verified_results.json"


def tex_name(value: str) -> str:
    value = str(value)
    for old, new in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        value = value.replace(old, new)
    return value


def compact(value: str, limit: int = 150) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def source_for(stem: str) -> Path | None:
    candidates = []
    for folder in SOURCE_DIRS:
        candidates.extend(folder.glob("*.py"))
    exact = [p for p in candidates if p.stem.lower() == stem.lower()]
    if exact:
        return sorted(exact, key=lambda p: str(p))[0]
    normalized = re.sub(r"[^a-z0-9]", "", stem.lower())
    close = [p for p in candidates if re.sub(r"[^a-z0-9]", "", p.stem.lower()) == normalized]
    return sorted(close, key=lambda p: str(p))[0] if close else None


def read_primary() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    if PRIMARY_CSV.exists():
        for row in csv.DictReader(PRIMARY_CSV.open(encoding="utf-8")):
            out.setdefault(row.get("module_name", "").lower(), []).append(row)
    return out


def read_verified() -> dict:
    if VERIFIED_JSON.exists():
        data = json.loads(VERIFIED_JSON.read_text(encoding="utf-8"))
        return data.get("modules", {})
    return {}


def registry_keys(source: str) -> list[str]:
    return re.findall(r"@(?:ATTENTION|COMPRESSOR)_REGISTRY\.register\(\s*[\"']([^\"']+)", source)


def classes(source: str) -> list[str]:
    return re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE)


def constructor_lines(source: str) -> list[str]:
    lines = source.splitlines()
    found = []
    in_init = False
    depth = 0
    for line in lines:
        if re.search(r"\bdef\s+__init__\s*\(", line):
            in_init = True
        if in_init:
            found.append(line.strip())
            depth += line.count("(") - line.count(")")
            if depth <= 0 and ":" in line:
                break
    return found[:30]


def evidence(source: str) -> list[str]:
    patterns = [
        (r"torch\.linalg\.svd|linalg\.svd|singular", "تجزیهٔ مقدار منفرد یا مسیر کم‌رتبه"),
        (r"lora|LoRA", "آداپتور LoRA و مقیاس‌گذاری رتبه‌ای"),
        (r"repeat_kv|num_kv_heads|grouped|GQA", "توجه با سرهای گروه‌بندی‌شده و بسط K/V"),
        (r"topk|top_k|top-k|sparse|block_size", "انتخاب Top-K یا محاسبهٔ تنک/بلاکی"),
        (r"rotary|rope|RoPE", "کدگذاری مکانی چرخشی"),
        (r"softmax", "نرمال‌سازی امتیازها با Softmax"),
        (r"scan|recurrence|state|decay|delta", "حالت بازگشتی، اسکن یا گیت زوال"),
        (r"landmark|nystrom|Nyström", "تقریب مبتنی بر نقاط لنگر/نیستروم"),
        (r"quant|fp8|int4|int8", "کوانتایزاسیون یا دقت عددی کاهش‌یافته"),
    ]
    return [label for pattern, label in patterns if re.search(pattern, source, re.IGNORECASE)]


def formula_block(features: list[str]) -> list[str]:
    blocks = [
        "برای ورودی $X\​\in\\mathbb{R}^{B\\times N\\times D}$، مسیر پایهٔ توجه با $Q=L_Q(X)$، $K=L_K(X)$ و $V=L_V(X)$ آغاز می‌شود؛ شکل معمول پس از تفکیک سرها $[B,H,N,d]$ است.",
        "امتیاز و خروجی توجه در صورت وجود مسیر Softmax به‌صورت $S=QK^{\\top}/\\sqrt{d}$ و $O=\\operatorname{Softmax}(S+M)V$ نوشته می‌شوند؛ $M$ ماسک علّی یا padding است و در کد باید مشخص شود که کدام‌یک اعمال می‌شود.",
    ]
    if "تجزیهٔ مقدار منفرد یا مسیر کم‌رتبه" in features:
        blocks.append("اگر کد SVD را اجرا کند، نگاشت وزن به‌صورت $W\\approx U_r\\Sigma_rV_r^{\\top}$ است و خطای بازسازی دقیقاً با $\\|W-W_r\\|_F$ سنجیده می‌شود؛ مقدار $r$ باید از سازنده یا تنظیمات واقعی خوانده شود، نه از متن عمومی.")
    if "آداپتور LoRA و مقیاس‌گذاری رتبه‌ای" in features:
        blocks.append("در مسیر LoRA، خروجی نگاشت $L(X)=W_0X+(\\alpha/r)BAX$ است؛ $W_0$ وزن پایه، $A$ و $B$ عوامل کم‌رتبه، و $\\alpha/r$ ضریب مقیاس‌گذاری است. فریز بودن $W_0$ باید از گزینهٔ واقعی ماژول بررسی شود.")
    if "توجه با سرهای گروه‌بندی‌شده و بسط K/V" in features:
        blocks.append("در GQA، تعداد سرهای پرس‌وجو $H$ از سرهای K/V یعنی $H_{kv}$ بیشتر است؛ بنابراین هر سر K/V با ضریب $G=H/H_{kv}$ تکرار می‌شود و سپس ضرب $QK^{\\top}$ انجام می‌گیرد.")
    if "انتخاب Top-K یا محاسبهٔ تنک/بلاکی" in features:
        blocks.append("در مسیر تنک، ابتدا امتیاز انتخاب برای بلوک‌ها یا توکن‌ها محاسبه می‌شود، سپس یک زیرمجموعهٔ $\\mathcal{I}_{topk}$ انتخاب و فقط $K_{\\mathcal{I}},V_{\\mathcal{I}}$ وارد Softmax می‌شوند. این انتخاب می‌تواند دقت و هزینه را هم‌زمان تغییر دهد و باید با نتایج CSV تفسیر شود.")
    if "کدگذاری مکانی چرخشی" in features:
        blocks.append("در RoPE، زوج‌بعدهای $Q$ و $K$ با زاویهٔ وابسته به موقعیت چرخانده می‌شوند؛ هدف، واردکردن اطلاعات نسبی موقعیت بدون افزودن ماتریس توجه جدید است.")
    if "حالت بازگشتی، اسکن یا گیت زوال" in features:
        blocks.append("در مسیر بازگشتی، حالت به‌شکل کلی $h_t=F(h_{t-1},x_t)$ به‌روزرسانی می‌شود. اگر کد گیت زوال داشته باشد، معمولاً $h_t=\\lambda_t h_{t-1}+u_t$ است؛ مقدار و دامنهٔ $\\lambda_t$ باید مطابق کد همان ماژول گزارش شود.")
    if "تقریب مبتنی بر نقاط لنگر/نیستروم" in features:
        blocks.append("در تقریب نیستروم، چند نقطهٔ لنگر به‌عنوان نمایندهٔ توالی انتخاب می‌شوند و ضرب توجه با ترکیب سه ضرب کوچک‌تر تقریب زده می‌شود؛ رتبه یا تعداد لنگر باید از تنظیمات واقعی ماژول استخراج شود.")
    if "کوانتایزاسیون یا دقت عددی کاهش‌یافته" in features:
        blocks.append("در مسیر کم‌دقت، تبدیل dtype یا کوانتایزاسیون بخشی از هزینهٔ حافظه را کم می‌کند، اما مقایسهٔ دقت باید با همان dtype و همان baseline انجام شود.")
    return blocks


def result_rows(stem: str, primary: dict[str, list[dict[str, str]]], verified: dict) -> list[dict[str, str]]:
    rows = primary.get(stem.lower(), [])
    if rows:
        return rows[:4]
    entry = verified.get(stem)
    if entry:
        out = []
        for model, item in entry.get("models", {}).items():
            out.append({
                "model_family": model,
                "dataset": "snapshot-specific",
                "status": str(item.get("status", "")),
                "mean_accuracy": str(item.get("accuracy_pct", "")),
                "f1_score": str(item.get("f1_score", "")),
                "peak_vram_mb": str(item.get("peak_vram_mb", "")),
                "throughput_tokens_sec": str(item.get("throughput_tokens_sec", "")),
                "source_file": str(item.get("source_file", "")),
            })
        return out[:4]
    return []


def render(stem: str, source_path: Path | None, source: str, primary: dict, verified: dict) -> str:
    display = stem
    keys = registry_keys(source)
    cls = classes(source)
    feats = evidence(source)
    source_rel = str(source_path.relative_to(ROOT)) if source_path else "Not found in Codes/"
    lines = [
        f"\\subsection{{کالبدشکافی دقیق ماژول \\lr{{{tex_name(display)}}}}}",
        "\\textbf{وضعیت مستندات:} این متن از فایل کد و snapshot نتیجهٔ واقعی تولید شده است؛ هر ادعایی که در منبع داده وجود نداشته باشد به‌عنوان Pending Verification باقی می‌ماند.",
        f"\\subsubsection*{{۱. نقش و هدف ماژول}}",
        f"ماژول \\lr{{{tex_name(display)}}} در چارچوب برای جایگزینی یا فشرده‌سازی یک مسیر توجه/فشرده‌ساز استفاده می‌شود. فایل مرجع آن \\lr{{{tex_name(source_rel)}}} است. این نگاشت بر اساس نام فایل انجام شده و برای تفاوت نام‌های رجیستری باید کلید واقعی رجیستری بررسی شود.",
        f"\\subsubsection*{{۲. کلاس‌ها و کلیدهای رجیستری}}",
        f"کلاس‌های پیدا‌شده: \\lr{{{tex_name(', '.join(cls) or 'در فایل پیدا نشد')}}}. کلیدهای رجیستری: \\lr{{{tex_name(', '.join(keys) or 'در فایل پیدا نشد')}}}.",
        "در نتیجه، نامی که در گزارش آزمایش می‌آید لزوماً نام کلاس پایتون نیست؛ برای بازتولید باید کلید رجیستری و مسیر تزریق هر دو ثبت شوند.",
        "\\subsubsection*{۳. فرمول‌بندی و معنای هر مرحله}",
    ]
    lines.extend(formula_block(feats))
    lines.extend([
        "\\subsubsection*{۴. مسیر تانسورها و ترتیب اجرای کد}",
        "۱) ورودی سه‌بعدی با شکل $[B,N,D]$ دریافت می‌شود. ۲) پروجکشن‌های تعریف‌شده در سازنده اجرا می‌شوند. ۳) در صورت وجود، reshape/transpose سرها، ماسک، RoPE یا GQA اعمال می‌شود. ۴) هستهٔ توجه/فشرده‌سازی اجرا می‌شود. ۵) سرها به شکل $[B,N,D]$ برگردانده و پروجکشن خروجی اعمال می‌گردد. این مسیر عمومی است؛ تفاوت‌های خاص این ماژول در شواهد زیر آمده‌اند.",
        f"شواهد محاسباتی استخراج‌شده از کد: \\lr{{{tex_name('; '.join(feats) or 'الگوی ویژه‌ای با جست‌وجوی ایستا پیدا نشد')}}}.",
        "\\subsubsection*{۵. تنظیمات و قرارداد سازنده}",
        "پارامترهای زیر از امضای واقعی \lr{__init__} استخراج شده‌اند؛ مقدارهای پیش‌فرض فقط وقتی گزارش می‌شوند که در همان امضا وجود داشته باشند:",
        "\\begin{itemize}",
    ])
    init = constructor_lines(source)
    for line in init[:24]:
        if line and not line.startswith("def __init__"):
            lines.append(f"\\item \\texttt{{{tex_name(line)}}}")
    if not init:
        lines.append("\\item امضای سازنده در فایل پیدا نشد؛ نیازمند بررسی دستی است.")
    lines.extend(["\\end{itemize}", "\\subsubsection*{۶. نتایج واقعی و شیوهٔ تفسیر}"])
    rows = result_rows(stem, primary, verified)
    if rows:
        lines.extend(["\\begin{center}", "\\begin{tabular}{lllllll}", "\\textbf{Model} & \\textbf{Task} & \\textbf{Status} & \\textbf{Acc.} & \\textbf{F1} & \\textbf{VRAM} & \\textbf{Tok/s} \\\\ \\hline"])
        for row in rows:
            lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
                tex_name(row.get("model_family", "")), tex_name(row.get("dataset", "")), tex_name(row.get("status", "")),
                tex_name(row.get("mean_accuracy", "")), tex_name(row.get("f1_score", "")), tex_name(row.get("peak_vram_mb", "")), tex_name(row.get("throughput_tokens_sec", ""))))
        lines.extend(["\\end{tabular}", "\\end{center}"])
        lines.append("منبع هر ردیف در snapshot نتیجه ثبت شده است؛ ردیف‌های صفر، FAIL، TIMEOUT یا SKIP موفقیت محسوب نمی‌شوند.")
    else:
        lines.append("برای این نام، رکورد قابل‌ردیابی در snapshotهای بررسی‌شده پیدا نشد؛ نتیجه: Pending Verification.")
    lines.extend([
        "\\subsubsection*{۷. محدودیت‌ها و خطاهای محتمل}",
        "این مستندات فقط رفتار قابل مشاهدهٔ کد را توضیح می‌دهند. تفاوت شکل پروجکشن‌ها، maskهای مدل، dtype، مسیر fallback، طول توالی و seed می‌تواند نتیجه را تغییر دهد. بنابراین پیچیدگی نظری، دقت، حافظه و سرعت جایگزین یکدیگر نیستند و باید از یک اجرای هم‌شرط گزارش شوند.",
        "\\subsubsection*{۸. مسیر بازتولید}",
        f"برای بازتولید، ابتدا کلید رجیستری ذکرشده را resolve کنید، سپس فایل \\lr{{{tex_name(source_rel)}}} را با همان config مدل اجرا کنید و ردیف CSV متناظر را با نام مدل، وظیفه، status و timestamp ثبت نمایید.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    primary = read_primary()
    verified = read_verified()
    inputs = []
    for line in (THESIS / "chapter_attentions.tex").read_text(encoding="utf-8").splitlines():
        match = re.search(r"\\input\{([^}]+)\}", line)
        if match:
            inputs.append(Path(match.group(1)).stem)
    generated = 0
    for stem in inputs:
        source_path = source_for(stem)
        source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path else ""
        target = MODULE_DIR / f"{stem}.tex"
        target.write_text(render(stem, source_path, source, primary, verified), encoding="utf-8")
        generated += 1
    print(f"generated={generated}")


if __name__ == "__main__":
    main()
