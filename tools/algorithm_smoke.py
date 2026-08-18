from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.pdf.extractor import extract_text
from src.pdf.parser import build_algorithm_sections, extract_diagnosis, parse_sections
from src.rag.pipeline import _group_by_chapter, stream_algorithm


SAMPLES = [
    ("KR1034 trauma teeth", "/tmp/KR1034.pdf", "Травма зубов"),
    ("KR1036 hirschsprung pdf", "/tmp/KR1036.pdf", "Болезнь Гиршпрунга"),
    ("KR286 type 1 diabetes", "/tmp/KR286_3.pdf", "Сахарный диабет 1 типа"),
    ("Hirschsprung db full_text", "/tmp/hirsch_full.txt", "Болезнь Гиршпрунга"),
    ("Retinopathy db full_text", "/tmp/retino_full.txt", "Ретинопатия недоношенных"),
]

BAD_PATTERNS = {
    "unknown_disease": re.compile(r"Неизвестное заболевание", re.I),
    "section_fragment_weeks": re.compile(r"^##\s+недели после", re.I | re.M),
    "generic_section_1": re.compile(r"^##\s+Раздел\s+1\b", re.I | re.M),
    "clindamycin_inverted": re.compile(
        r"клиндамицин[^\n.]{0,80}противопоказан[^\n.]{0,80}бета-лактам",
        re.I,
    ),
    "laser_peripheral_inverted": re.compile(
        r"ЛКС не показан[^\n.]{0,120}периферическ",
        re.I,
    ),
}


def _read_source(path: str) -> str:
    source = Path(path)
    if source.suffix.lower() == ".pdf":
        return extract_text(str(source))
    return source.read_text(encoding="utf-8")


def _safe_name(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return value or "sample"


def _first_heading(markdown: str) -> str:
    match = re.search(r"(?m)^##\s+(.+)$", markdown)
    return match.group(1).strip() if match else ""


def _bad_headings(markdown: str) -> list[str]:
    bad = []
    for heading in re.findall(r"(?m)^##\s+(.+)$", markdown):
        clean = heading.strip()
        lower = clean.lower()
        if lower.startswith(("недели ", "дни ", "дня ", "после ")):
            bad.append(clean)
    return bad


def _checks(markdown: str, diagnosis: str | None, expected: str) -> list[str]:
    issues = []
    if not diagnosis:
        issues.append("missing_diagnosis")
    elif expected and expected.lower() not in diagnosis.lower() and diagnosis.lower() not in expected.lower():
        issues.append(f"diagnosis_mismatch: got={diagnosis!r} expected~={expected!r}")

    if len(markdown.strip()) < 3000:
        issues.append(f"short_algorithm: {len(markdown.strip())} chars")

    first = _first_heading(markdown)
    if not first:
        issues.append("missing_h2_heading")
    elif first[:1].islower():
        issues.append(f"first_heading_starts_lowercase: {first!r}")

    for name, pattern in BAD_PATTERNS.items():
        if pattern.search(markdown):
            issues.append(f"bad_pattern:{name}")

    for heading in _bad_headings(markdown):
        issues.append(f"bad_heading:{heading[:120]}")

    return issues


def run_sample(label: str, path: str, expected: str, out_dir: Path, generate: bool) -> dict:
    started = time.time()
    full_text = _read_source(path)
    filename = Path(path).name if Path(path).suffix.lower() == ".pdf" else None
    diagnosis, mkb_code = extract_diagnosis(
        "",
        full_text,
        filename=filename,
        allow_llm_fallback=False,
    )
    sections = parse_sections(full_text)
    algorithm_sections = build_algorithm_sections(full_text)
    chapters = _group_by_chapter(full_text)

    markdown = ""
    output_path = None
    if generate:
        markdown = "".join(stream_algorithm(full_text, diagnosis or expected or label))
        output_path = out_dir / f"{_safe_name(label)}.md"
        output_path.write_text(markdown, encoding="utf-8")

    issues = _checks(markdown, diagnosis, expected) if generate else []
    return {
        "label": label,
        "path": path,
        "expected": expected,
        "diagnosis": diagnosis,
        "mkb_code": mkb_code,
        "text_chars": len(full_text),
        "section_count": len(sections),
        "algorithm_section_count": len(algorithm_sections),
        "chapter_count": len(chapters),
        "first_chapter": chapters[0][0] if chapters else "",
        "algorithm_chars": len(markdown),
        "first_heading": _first_heading(markdown) if markdown else "",
        "issues": issues,
        "output_path": str(output_path) if output_path else None,
        "elapsed_sec": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/clinical_ai_smoke")
    parser.add_argument("--report", default="/tmp/clinical_ai_smoke/report.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, path, expected in SAMPLES:
        if not Path(path).exists():
            results.append({
                "label": label,
                "path": path,
                "expected": expected,
                "issues": ["missing_input"],
            })
            continue
        print(f"RUN {label}", flush=True)
        try:
            result = run_sample(label, path, expected, out_dir, args.generate)
        except Exception as exc:
            result = {
                "label": label,
                "path": path,
                "expected": expected,
                "issues": [f"exception:{type(exc).__name__}: {exc}"],
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    report = {"generated": args.generate, "results": results}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {args.report}", flush=True)


if __name__ == "__main__":
    main()
