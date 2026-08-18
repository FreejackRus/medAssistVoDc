from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx

from src.pdf.extractor import extract_text
from src.pdf.parser import build_algorithm_sections, extract_diagnosis, parse_sections
from src.rag.chunker import chunk_sections
from src.rag.pipeline import _group_by_chapter, stream_algorithm, stream_rag_answer
from src.rag.vector_store import add_chunks, delete_document_chunks


API_URL = "https://apicr.minzdrav.gov.ru/api.ashx"
CATALOG_URL = f"{API_URL}?op=GetJsonClinrecsFilterV2"
USER_AGENT = "Mozilla/5.0 (compatible; ClinicalAIRegression/1.0)"
CHAT_QUESTION = "Про что документ и как в целом лечить пациента? Ответь кратко, не больше 6 пунктов."

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


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-zА-Яа-яЁё0-9._-]+", "_", value).strip("_")
    return value[:120] or "sample"


def fetch_recommendations(
    limit: int,
    offset: int = 0,
    exclude_codes: Optional[set[str]] = None,
) -> list[dict]:
    exclude_codes = exclude_codes or set()
    page_size = max(limit + offset + len(exclude_codes) + 20, 40)
    payload = {
        "filters": [
            {
                "fieldName": "status",
                "filterType": 1,
                "filterValueType": 2,
                "value1": 0,
                "value2": "",
                "values": [],
            }
        ],
        "sortOption": {"fieldName": "publishdate", "sortType": 2},
        "pageSize": page_size,
        "currentPage": 1,
        "useANDoperator": True,
        "columns": [],
    }
    with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        response = client.post(CATALOG_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
    items = [
        item
        for item in data.get("Data", [])
        if str(item.get("CodeVersion") or item.get("Id") or "") not in exclude_codes
    ]
    return items[offset:offset + limit]


def download_pdf(code_version: str, target: Path) -> None:
    url = f"{API_URL}?id={code_version}&op=GetClinrecPdf"
    with httpx.Client(timeout=90.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        content = response.content
    if len(content) < 1000 or not content.startswith(b"%PDF-"):
        raise RuntimeError(f"downloaded content is not a PDF: {len(content)} bytes")
    target.write_bytes(content)


def first_heading(markdown: str) -> str:
    match = re.search(r"(?m)^##\s+(.+)$", markdown)
    return match.group(1).strip() if match else ""


def has_bad_heading(markdown: str) -> bool:
    for heading in re.findall(r"(?m)^##\s+(.+)$", markdown):
        if heading.strip().lower().startswith(("недели ", "дни ", "дня ", "после ")):
            return True
    return False


def diagnosis_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    stopwords = {
        "болезнь",
        "синдром",
        "заболевание",
        "состояние",
        "типа",
        "тип",
        "детей",
        "подростков",
    }
    tokens = re.findall(r"[А-Яа-яЁёA-Za-z]{4,}", value.lower())
    return [token for token in tokens if token not in stopwords]


def diagnosis_matches_title(diagnosis: str | None, title: str) -> bool:
    if not diagnosis:
        return False
    diagnosis_lower = diagnosis.lower()
    title_lower = title.lower()
    if diagnosis_lower in title_lower or title_lower in diagnosis_lower:
        return True
    title_tokens = diagnosis_tokens(title)
    if not title_tokens:
        return True
    matches = sum(1 for token in title_tokens if token in diagnosis_lower)
    return matches >= max(1, min(2, len(title_tokens)))


def token_in_text(token: str, text: str) -> bool:
    if token in text:
        return True
    if len(token) >= 7 and token[:-1] in text:
        return True
    if len(token) >= 8 and token[:-2] in text:
        return True
    return False


def check_text(markdown: str, diagnosis: str | None, *, is_chat: bool = False) -> list[str]:
    issues = []
    if len(markdown.strip()) < (400 if is_chat else 3000):
        issues.append(f"short_text:{len(markdown.strip())}")

    if not is_chat:
        heading = first_heading(markdown)
        if not heading:
            issues.append("missing_h2")
        elif heading[:1].islower():
            issues.append(f"lowercase_first_heading:{heading}")
        if has_bad_heading(markdown):
            issues.append("bad_fragment_heading")

    for name, pattern in BAD_PATTERNS.items():
        if pattern.search(markdown):
            issues.append(f"bad_pattern:{name}")

    tokens = diagnosis_tokens(diagnosis)
    markdown_lower = markdown.lower()
    if tokens and not any(token_in_text(token, markdown_lower) for token in tokens[:4]):
        issues.append("diagnosis_terms_missing")

    return issues


def run_one(item: dict, out_dir: Path) -> dict:
    started = time.time()
    code_version = str(item.get("CodeVersion") or item.get("Id") or "")
    title = str(item.get("Name") or code_version)
    api_mkb = ""
    mkbs = item.get("Mkbs") or []
    if mkbs:
        api_mkb = str(mkbs[0].get("MkbCode") or "")
    elif item.get("Code") is not None:
        api_mkb = str(item.get("Code"))

    prefix = safe_name(f"{code_version}_{title}")
    pdf_path = out_dir / "pdf" / f"{prefix}.pdf"
    algorithm_path = out_dir / "algorithms" / f"{prefix}.md"
    chat_path = out_dir / "chat" / f"{prefix}.md"
    document_id = f"minzdrav-reg-{code_version}"

    result = {
        "code_version": code_version,
        "title": title,
        "api_mkb": api_mkb,
        "publishdate": item.get("PublishDateStr"),
        "age_group": item.get("AgeCategoryStr"),
        "issues": [],
    }

    try:
        download_pdf(code_version, pdf_path)
        full_text = extract_text(str(pdf_path))
        diagnosis, mkb_code = extract_diagnosis(
            "",
            full_text,
            filename=f"{title}.pdf",
            allow_llm_fallback=False,
        )
        sections = parse_sections(full_text)
        algorithm_sections = build_algorithm_sections(full_text)
        chapters = _group_by_chapter(full_text)

        delete_document_chunks(document_id)
        chunks = chunk_sections(sections, document_id=document_id)
        chunk_count = add_chunks(chunks, document_id=document_id)

        algorithm = "".join(stream_algorithm(full_text, diagnosis or title))
        algorithm_path.write_text(algorithm, encoding="utf-8")

        chat = "".join(stream_rag_answer(CHAT_QUESTION, document_id, diagnosis or title, []))
        chat_path.write_text(chat, encoding="utf-8")

        result.update(
            {
                "diagnosis": diagnosis,
                "mkb_code": mkb_code,
                "text_chars": len(full_text),
                "section_count": len(sections),
                "algorithm_section_count": len(algorithm_sections),
                "chapter_count": len(chapters),
                "first_chapter": chapters[0][0] if chapters else "",
                "chunk_count": chunk_count,
                "algorithm_chars": len(algorithm),
                "algorithm_first_heading": first_heading(algorithm),
                "chat_chars": len(chat),
                "algorithm_path": str(algorithm_path),
                "chat_path": str(chat_path),
            }
        )
        result["issues"].extend(check_text(algorithm, diagnosis, is_chat=False))
        result["issues"].extend(f"chat:{issue}" for issue in check_text(chat, diagnosis, is_chat=True))
        if not diagnosis_matches_title(diagnosis, title):
            result["issues"].append("diagnosis_title_mismatch")
    except Exception as exc:
        result["issues"].append(f"exception:{type(exc).__name__}:{exc}")
    finally:
        try:
            delete_document_chunks(document_id)
        except Exception as exc:
            result["issues"].append(f"cleanup:{type(exc).__name__}:{exc}")

    result["elapsed_sec"] = round(time.time() - started, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--exclude-codes", default="")
    parser.add_argument("--out-dir", default="/tmp/minzdrav_regression")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for child in ("pdf", "algorithms", "chat"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)

    exclude_codes = {
        code.strip()
        for code in args.exclude_codes.split(",")
        if code.strip()
    }
    recs = fetch_recommendations(args.limit, args.offset, exclude_codes)
    results = []
    for index, item in enumerate(recs, 1):
        print(f"RUN {index}/{len(recs)} {item.get('CodeVersion')} {item.get('Name')}", flush=True)
        result = run_one(item, out_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    report = {
        "limit": args.limit,
        "offset": args.offset,
        "exclude_codes": sorted(exclude_codes),
        "count": len(results),
        "question": CHAT_QUESTION,
        "results": results,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)


if __name__ == "__main__":
    main()
