from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)

SECTION_PATTERNS = [
    re.compile(r"^([1-7](?:\.\d+)*)\s+([^\n]+)", re.MULTILINE),
    re.compile(r"^([IVX]+)\.\s+([^\n]+)", re.MULTILINE),
    re.compile(r"^([1-7])\.\s+([А-ЯЁ][^\n]+)", re.MULTILINE),
    re.compile(r"^([А-Я][а-я]+(?:\s+[а-я]+)*)\s*\n", re.MULTILINE),
    re.compile(r"^([А-ЯЁ]{2,}(?:\s+[А-ЯЁ]{2,})*)\s*\n(?=[А-ЯЁа-яё])", re.MULTILINE),
]

MEDICAL_KEYWORDS = [
    "определение", "этиология", "патогенез", "классификация", "диагностика",
    "лечение", "терапия", "профилактика", "прогноз", "рекомендации",
    "показания", "противопоказания", "дозировка", "мониторинг", "осложнения",
]

MKB_PATTERNS = [
    re.compile(r"МКБ[- ]?10?[:\s]*([A-Z]\d{2}(?:\.\d{1,2})?)", re.I),
    re.compile(r"код[:\s]*([A-Z]\d{2}(?:\.\d{1,2})?)", re.I),
    re.compile(r"\b([A-Z]\d{2}\.\d{1,2})\b"),
]
VALID_MKB_PATTERN = re.compile(r"^[A-Z]\d{2}(?:\.\d{1,2})?$")

DIAGNOSIS_PATTERNS = [
    re.compile(r"клинические\s+рекомендации\s*\n\s*(.+?)(?:\n|$)", re.I),
    re.compile(r"клинические\s+рекомендации\s+(?:по\s+)?(.+?)(?:\n|$)", re.I),
    re.compile(r"протокол\s+ведения\s+больных\s+(.+?)(?:\n|$)", re.I),
    re.compile(r"стандарт\s+медицинской\s+помощи\s+(?:при\s+)?(.+?)(?:\n|$)", re.I),
    re.compile(r"алгоритм\s+(?:диагностики\s+и\s+)?лечения\s+(.+?)(?:\n|$)", re.I),
    re.compile(r"методические\s+рекомендации\s+(?:по\s+)?(.+?)(?:\n|$)", re.I),
    re.compile(
        r"^([А-ЯЁ][А-Яа-яЁё0-9\s()/,-]+(?:болезнь|синдром|заболевание|патология|недостаточность|"
        r"гипертензия|диабет|астма|пневмония|инфаркт|стенокардия|аритмия|"
        r"инфекция|гепатит|туберкулёз|туберкулез|анемия|остеопороз|эпилепсия|"
        r"склероз|ожирение|артрит|дерматит|псориаз)[а-яё\s]*)",
        re.M,
    ),
]

DIAGNOSIS_TERM_PATTERN = (
    r"(?:болезнь|синдром|заболевание|патология|недостаточность|гипертензия|диабет|астма|"
    r"пневмония|инфаркт|стенокардия|аритмия|нефропатия|ретинопатия|нейропатия|"
    r"инфекция|гепатит|туберкулёз|туберкулез|анемия|лейкоз|лимфома|остеопороз|"
    r"эпилепсия|склероз|ожирение|подагра|колит|гастрит|панкреатит|цирроз|"
    r"менингит|энцефалит|дерматит|псориаз|артрит|травм\w*|перелом|вывих|"
    r"низкоросл\w*|гиперинсулинизм|эпидермолиз|герпес|митохондриаль\w*)"
)

POTENTIAL_SECTION_PATTERN = re.compile(r"(?m)^\s*([1-7](?:\.\d+){0,2})\.?\s+([^\n]+)")
OUTLINE_ENTRY_PATTERN = re.compile(
    r"(?m)^\s*([1-7](?:\.\d+){0,2})\.?\s+([^\n]+?)(?:\s\.{2,}\s*\d+)?\s*$"
)
MAIN_CONTENT_START_PATTERNS = (
    re.compile(r"\f\s*1\.\s+Краткая\s+информация", re.I),
    re.compile(r"(?m)^1\.\s+Краткая\s+информация", re.I),
    re.compile(r"(?m)^\s*1\s*\n\s*Краткая\s+информация", re.I),
    re.compile(r"(?m)^\s*1\s+Краткая\s+информация", re.I),
    re.compile(r"(?m)^\s*Краткая информация по заболеванию", re.I),
)
# TOC entries can have dot leaders or plain page numbers on wrapped lines.
_TOC_LINE_PATTERN = re.compile(r"\.{2,}\s*\d+\s*$", re.MULTILINE)
APPENDIX_START_PATTERN = re.compile(
    r"(?m)^Приложение\s+[AА]\d(?:[-–]\d+)?\.[ \t]+[А-ЯЁA-Z][^\n]{5,}"
)
REFERENCES_START_PATTERN = re.compile(r"(?mi)^Список\s+литературы[ \t]*$")

TOP_LEVEL_SECTION_KEYWORDS = (
    "краткая информация",
    "диагностика",
    "лечение",
    "реабилитация",
    "санаторно",
    "профилактика",
    "диспансерное наблюдение",
    "организация оказания",
    "дополнительная информация",
    "особенности",
    "осложнения",
)

TOP_LEVEL_SECTION_PREFIXES = (
    "краткая информация",
    "диагностика",
    "лечение",
    "медицинская реабилитация",
    "реабилитация",
    "профилактика",
    "диспансерное наблюдение",
    "организация оказания",
    "дополнительная информация",
)

TOC_SECTION_HINTS = (
    "краткая информация",
    "диагностика",
    "лечение",
    "медицинская реабилитация",
    "профилактика",
    "организация оказания",
    "дополнительная информация",
    "список сокращений",
    "термины и определения",
    "приложение",
)

EXCLUDED_SECTION_KEYWORDS = (
    "список сокращений",
    "термины и определения",
    "критерии оценки качества",
    "проведено",
    "да/нет",
    "опросник",
    "mini-cog",
    "рисование часов",
    "приложение",
    "анкета",
    "шкала",
    "опрос",
)

DEFINITION_SECTION_PATTERN = re.compile(
    r"(?ims)^\s*1\.1\.?\s*(?:\n\s*)?Определение[^\n]*\n+(.*?)(?=^\s*1\.2\.?\s*(?:\n\s*)?[А-ЯЁ]|\Z)"
)


def _is_valid_section_title(title: str) -> bool:
    if not title or len(title.strip()) < 3:
        return False
    t = title.lower().strip()
    if re.match(r"^\d{4}\s*г\.?\s*", t):
        return False
    if title.isupper() and len(title) < 10:
        return False
    if any(w in t for w in ["%", "процент", "превысила", "составила"]):
        return False
    if any(kw in t for kw in MEDICAL_KEYWORDS):
        return True
    if re.match(r"^\d+\.?\d*\s+[а-яё]", t):
        return True
    if re.match(r"^[А-ЯЁ][а-яё]", title) and len(title) > 5:
        return True
    return False


def _clean_section_title(title: str) -> str:
    title = re.sub(r"\.{2,}\s*\d+\s*$", "", title)
    title = re.sub(r"\.{2,}\s*$", "", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title


def _normalize_title_key(title: str) -> str:
    title = _clean_section_title(title).lower().replace("ё", "е")
    title = re.sub(r"[^\w\s/-]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _is_top_level_section_title(title: str) -> bool:
    normalized = _normalize_title_key(title)
    return any(normalized.startswith(prefix) for prefix in TOP_LEVEL_SECTION_PREFIXES)


def _normalize_diagnosis_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\"\'()\[\]{}]", "", name)
    name = re.sub(r"\s*(?:—|–|-|:)\s*$", "", name)
    return name.strip(" .,;:")


def _diagnosis_candidate_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    name = re.sub(r"^(?:КР|KR)?\s*\d+(?:[_-]\d+)?[_\s-]+", "", name, flags=re.I)
    name = _normalize_diagnosis_name(name.replace("_", " "))
    if _looks_like_filename_diagnosis(name):
        return name
    return None


def _normalize_mkb_code(code: str | None) -> str | None:
    if not code:
        return None
    match = re.search(r"[A-Z]\d{2}(?:\.\d{1,2})?", code.upper())
    return match.group(0) if match else None


def _normalize_match_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_diagnosis(name: str) -> bool:
    if len(name) < 10 or len(name) > 160:
        return False
    lower = name.lower()
    if any(
        junk in lower
        for junk in [
            "утвержден",
            "ассоциаци",
            "оглавление",
            "список сокращений",
            "подозрен",
            "исследование",
            "проводят",
            "проводится",
            "рекомендуется",
            "пациент",
            "является",
            "представляет",
            "составляющ",
            "гетерогенным",
        ]
    ):
        return False
    return bool(re.search(DIAGNOSIS_TERM_PATTERN, lower, re.I))


def _looks_like_filename_diagnosis(name: str) -> bool:
    if len(name) < 5 or len(name) > 180:
        return False
    lower = name.lower()
    if any(
        junk in lower
        for junk in [
            "клинические рекомендации",
            "алгоритм",
            "приложение",
            "оглавление",
            "список сокращений",
            "неизвест",
            "исследование",
            "пациент",
            "является",
            "представляет",
        ]
    ):
        return False
    if re.fullmatch(r"(?:кр|kr)?\s*\d+(?:[_-]\d+)?", lower):
        return False
    return bool(re.search(r"[А-Яа-яЁё]{4,}", name))


def _diagnosis_supported_by_text(name: str, text: str) -> bool:
    normalized_name = _normalize_match_text(name)
    normalized_text = _normalize_match_text(text)
    if not normalized_name or not normalized_text:
        return False
    if normalized_name in normalized_text:
        return True

    stopwords = {
        "болезнь",
        "синдром",
        "заболевание",
        "патология",
        "недостаточность",
        "типа",
        "тип",
        "состояние",
    }
    tokens = [
        token
        for token in normalized_name.split()
        if len(token) >= 4 and token not in stopwords
    ]
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in normalized_text)
    return matches >= min(2, len(tokens))


def _mkb_supported_by_text(code: str, text: str) -> bool:
    normalized = _normalize_mkb_code(code)
    if not normalized:
        return False
    compact_text = re.sub(r"\s+", "", text.upper())
    return normalized in compact_text or normalized.replace(".", "") in compact_text


def _line_has_toc_section_hint(line: str) -> bool:
    normalized = _normalize_title_key(line)
    normalized = re.sub(r"^\d+(?:\.\d+){0,2}\s+", "", normalized)
    return any(normalized.startswith(hint) for hint in TOC_SECTION_HINTS)


def _looks_like_toc_candidate(full_text: str, start: int, end: int) -> bool:
    before = full_text[max(0, start - 700):start].lower()
    lookahead = full_text[end:end + 2200]
    if "оглавление" not in before and "содержание" not in before:
        return False
    if _TOC_LINE_PATTERN.search(lookahead):
        return True

    lines = [line.strip() for line in lookahead.splitlines()[:80]]
    toc_entries = 0
    for index, line in enumerate(lines):
        if not line or not _line_has_toc_section_hint(line):
            continue
        window = lines[index:index + 5]
        if re.search(r"\s\d{1,3}$", line) or any(
            re.fullmatch(r"\d{1,3}", candidate) for candidate in window[1:]
        ):
            toc_entries += 1

    return toc_entries >= 4


def _get_main_content_bounds(full_text: str) -> tuple[int, int]:
    starts: list[int] = []
    fallback_starts: list[int] = []
    for pattern in MAIN_CONTENT_START_PATTERNS:
        for match in pattern.finditer(full_text):
            fallback_starts.append(match.start())
            if _looks_like_toc_candidate(full_text, match.start(), match.end()):
                continue
            starts.append(match.start())

    if not starts and not fallback_starts:
        return 0, len(full_text)

    # Clinical recommendations often repeat the chapter title in the table of
    # contents. Use the latest plausible chapter start to avoid indexing TOC
    # entries as the body text.
    start = max(starts or fallback_starts)
    end_candidates = [
        match.start()
        for pattern in (REFERENCES_START_PATTERN, APPENDIX_START_PATTERN)
        for match in pattern.finditer(full_text)
        if match.start() > start
    ]
    end = min(end_candidates, default=len(full_text))
    return start, end


def _prepare_main_content_text(full_text: str) -> str:
    start, end = _get_main_content_bounds(full_text)
    return full_text[start:end].strip()


def _extract_outline_map(full_text: str) -> dict[str, list[str]]:
    start, _ = _get_main_content_bounds(full_text)
    if start <= 0:
        return {}

    outline: dict[str, list[str]] = {}
    front_matter = full_text[:start]
    for match in OUTLINE_ENTRY_PATTERN.finditer(front_matter):
        number = match.group(1)
        title = _clean_section_title(match.group(2))
        if re.match(r"^\d", title):
            continue
        normalized = _normalize_title_key(title)
        if len(normalized) < 5:
            continue
        outline.setdefault(number, [])
        if normalized not in outline[number]:
            outline[number].append(normalized)
    return outline


def _matches_outline(number: str, title: str, outline: dict[str, list[str]]) -> bool:
    normalized = _normalize_title_key(title)
    for candidate in outline.get(number, []):
        shorter, longer = sorted((normalized, candidate), key=len)
        if len(shorter) < 10:
            continue
        if longer.startswith(shorter) or shorter in longer:
            return True
    return False


def _is_probable_section_heading(number: str, title: str, outline: dict[str, list[str]]) -> bool:
    if not _is_valid_section_title(title):
        return False

    normalized = _normalize_title_key(title)
    if not normalized or re.match(r"^\d", normalized):
        return False

    if _matches_outline(number, title, outline):
        return True

    if "." not in number and outline.get(number):
        return False

    if "." in number:
        return (
            len(normalized.split()) <= 14
            and not normalized.endswith((",", ":", ";"))
            and " да нет " not in f" {normalized} "
        )

    return _is_top_level_section_title(title)


def _consume_wrapped_title(text: str, title_end: int, initial_title: str) -> tuple[str, int]:
    title_parts = [initial_title.strip()]
    cursor = title_end
    if cursor < len(text) and text[cursor] == "\n":
        cursor += 1

    while cursor < len(text):
        line_end = text.find("\n", cursor)
        if line_end == -1:
            line_end = len(text)
        line = text[cursor:line_end]
        stripped = line.strip()

        if not stripped or re.match(r"^\d+\s*$", stripped):
            break
        if POTENTIAL_SECTION_PATTERN.match(stripped):
            break
        if len(stripped) > 120:
            break
        if stripped[0].islower():
            title_parts.append(stripped)
            cursor = line_end
            if cursor < len(text) and text[cursor] == "\n":
                cursor += 1
            continue
        break

    return _clean_section_title(" ".join(title_parts)), cursor


def _find_diagnosis_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    definition_match = DEFINITION_SECTION_PATTERN.search(text)
    if definition_match:
        definition_body = definition_match.group(1).strip()
        for line in definition_body.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.split(r"\s+[—–-]\s+", line, maxsplit=1)[0]
            line = re.split(r"\s+\(", line, maxsplit=1)[0]
            name = _normalize_diagnosis_name(line)
            if _looks_like_diagnosis(name):
                candidates.append(name)
                break

    for pattern in DIAGNOSIS_PATTERNS:
        for match in pattern.finditer(text):
            name = _normalize_diagnosis_name(match.group(1))
            if _looks_like_diagnosis(name):
                candidates.append(name)

    generic_match = re.search(
        rf"(?im)^\s*([А-ЯЁ][А-Яа-яЁё0-9\s,/()-]{{8,120}}?{DIAGNOSIS_TERM_PATTERN}(?:\s+\d+\s+типа|\s+\d+\s+стадии|\s+[IVX]+)?)\s*(?:\(|—|–|-|:|$)",
        text,
    )
    if generic_match:
        name = _normalize_diagnosis_name(generic_match.group(1))
        if _looks_like_diagnosis(name):
            candidates.append(name)

    return candidates


def extract_definition(text: str) -> str:
    for title, body in build_algorithm_sections(text):
        if re.match(r"^1\.1\b", title) and "определение" in title.lower():
            return body.strip()

    definition_match = DEFINITION_SECTION_PATTERN.search(_prepare_main_content_text(text))
    if not definition_match:
        return ""
    return definition_match.group(1).strip()


def _extract_definition_excerpt(text: str) -> str:
    return extract_definition(text)[:2500]


def _build_diagnosis_fallback_context(
    first_page_text: str,
    full_text: str | None = None,
) -> str:
    parts = [f"<first_page>\n{first_page_text[:2500].strip()}\n</first_page>"]
    if full_text:
        main_text = _prepare_main_content_text(full_text)
        definition_excerpt = _extract_definition_excerpt(main_text)
        if definition_excerpt:
            parts.append(f"<definition_section>\n{definition_excerpt}\n</definition_section>")
        elif main_text:
            parts.append(f"<main_excerpt>\n{main_text[:2500].strip()}\n</main_excerpt>")
    return "\n\n".join(parts)


_diagnosis_llm = None


def _llm_extract_diagnosis(text: str) -> tuple[str | None, str | None]:
    """Fallback: ask LLM to extract diagnosis name and ICD-10 code."""
    global _diagnosis_llm
    try:
        if _diagnosis_llm is None:
            from src.llm.client import OllamaClient
            _diagnosis_llm = OllamaClient()
        response = _diagnosis_llm.chat(
            [
                {
                    "role": "system",
                    "content": "Извлеки название заболевания и код МКБ-10 из текста. "
                    'Ответь ТОЛЬКО JSON: {"name": "...", "mkb": "..."}. '
                    "Если не найдено, поставь null.",
                },
                {"role": "user", "content": text[:3000]},
            ],
            temperature=0.0,
            num_predict=150,
        )
        import json

        raw = response.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        name = data.get("name")
        mkb = _normalize_mkb_code(data.get("mkb"))
        if name and len(name) > 5:
            name = _normalize_diagnosis_name(name)
        else:
            name = None
        log.info("LLM fallback extracted diagnosis: name=%s, mkb=%s", name, mkb)
        return name, mkb
    except Exception as e:
        log.warning("LLM diagnosis fallback failed: %s", e)
        return None, None


def extract_diagnosis(
    first_page_text: str,
    full_text: str | None = None,
    filename: str | None = None,
    *,
    allow_llm_fallback: bool = True,
) -> tuple[str | None, str | None]:
    """Return (diagnosis_name, mkb_code) from PDF text."""
    mkb_code = None
    search_sources: list[str] = []
    if full_text:
        main_text = _prepare_main_content_text(full_text)
        if main_text:
            search_sources.append(main_text[:80000])
        search_sources.append(full_text[:80000])
    search_sources.append(first_page_text)

    for source in search_sources:
        for pattern in MKB_PATTERNS:
            matches = pattern.findall(source)
            if matches:
                mkb_code = matches[0].upper()
                break
        if mkb_code:
            break

    diagnosis_name = _diagnosis_candidate_from_filename(filename)
    for source in search_sources:
        if diagnosis_name:
            break
        candidates = _find_diagnosis_candidates(source)
        if candidates:
            diagnosis_name = candidates[0]
            break

    if not diagnosis_name and full_text:
        for line in full_text.split("\n")[:200]:
            line = _normalize_diagnosis_name(line.strip())
            if _looks_like_diagnosis(line):
                diagnosis_name = line
                break

    if allow_llm_fallback and (not diagnosis_name or not mkb_code):
        fallback_context = _build_diagnosis_fallback_context(first_page_text, full_text)
        llm_name, llm_mkb = _llm_extract_diagnosis(fallback_context)
        support_text = full_text or first_page_text

        if not diagnosis_name and llm_name and _looks_like_diagnosis(llm_name):
            if _diagnosis_supported_by_text(llm_name, support_text):
                diagnosis_name = llm_name

        if not mkb_code and llm_mkb and VALID_MKB_PATTERN.match(llm_mkb):
            if _mkb_supported_by_text(llm_mkb, support_text):
                mkb_code = llm_mkb

    return diagnosis_name, mkb_code


def parse_sections(full_text: str) -> dict[str, str]:
    """Parse text into sections dict. Falls back to keyword-based splitting."""
    sections: dict[str, str] = {}

    working_text = _prepare_main_content_text(full_text)
    outline = _extract_outline_map(full_text)
    headings: list[tuple[int, int, str, str]] = []

    for match in POTENTIAL_SECTION_PATTERN.finditer(working_text):
        number = match.group(1).strip()
        title, body_start = _consume_wrapped_title(working_text, match.end(), match.group(2))
        if _is_probable_section_heading(number, title, outline):
            headings.append((match.start(), body_start, number, title))

    for idx, (_, body_start, number, title) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(working_text)
        body = working_text[body_start:next_start].strip()
        body = re.sub(r"(?m)^\s*\d+\s*$", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body or len(body) <= 50:
            continue

        key = f"{number} {title}".strip()
        existing = sections.get(key)
        if existing is None or len(body) > len(existing):
            sections[key] = body

    if not sections:
        for pattern in SECTION_PATTERNS:
            splits = pattern.split(working_text)
            if len(splits) > 3:
                for i in range(1, len(splits), 3):
                    if i + 2 < len(splits):
                        number = splits[i].strip()
                        title = _clean_section_title(splits[i + 1].strip())
                        body = splits[i + 2].strip()
                        if _is_valid_section_title(title) and body and len(body) > 50:
                            key = f"{number} {title}".strip() if number else title
                            existing = sections.get(key)
                            if existing is None or len(body) > len(existing):
                                sections[key] = body
                if sections:
                    break

    if not sections:
        sections["guidelines"] = working_text or full_text

    # Try keyword fallback if only one blob
    if len(sections) == 1 and "guidelines" in sections:
        text = sections["guidelines"]
        keywords = [
            ("Определение", r"(определение|дефиниция)"),
            ("Этиология", r"(этиология|причины|факторы риска)"),
            ("Патогенез", r"(патогенез|механизм)"),
            ("Классификация", r"(классификация|типы|виды)"),
            ("Диагностика", r"(диагностика|диагноз|обследование)"),
            ("Лечение", r"(лечение|терапия|препараты)"),
            ("Профилактика", r"(профилактика|предупреждение)"),
            ("Прогноз", r"(прогноз|исход)"),
        ]
        kw_sections: dict[str, str] = {}
        all_patterns = "|".join(p[1] for p in keywords)
        for name, pat in keywords:
            m = re.search(f"({pat}.*?)(?={all_patterns}|$)", text, re.I | re.DOTALL)
            if m:
                kw_sections[name] = m.group(1).strip()
        if kw_sections:
            sections = kw_sections

    log.info("Parsed %d sections", len(sections))
    return sections


def build_algorithm_sections(full_text: str) -> list[tuple[str, str]]:
    """Select core clinical recommendation sections for algorithm generation."""
    sections = parse_sections(full_text)
    selected_by_number: dict[str, tuple[str, str]] = {}
    top_level_candidates: dict[str, tuple[str, str]] = {}

    for title, body in sections.items():
        clean_title = _clean_section_title(title)
        lower_title = clean_title.lower()
        number_match = re.match(r"^([1-7](?:\.\d+){0,2})\b", clean_title)
        if not number_match:
            continue
        if any(keyword in lower_title for keyword in EXCLUDED_SECTION_KEYWORDS):
            continue
        number = number_match.group(1)
        is_top_level = "." not in number

        if is_top_level and not any(keyword in lower_title for keyword in TOP_LEVEL_SECTION_KEYWORDS):
            continue

        entry = (clean_title, body.strip())
        target = top_level_candidates if is_top_level else selected_by_number
        existing = target.get(number)
        if existing is None or len(entry[1]) > len(existing[1]):
            target[number] = entry

    for number, entry in top_level_candidates.items():
        if not any(key.startswith(f"{number}.") for key in selected_by_number):
            selected_by_number[number] = entry

    selected = list(selected_by_number.values())

    def sort_key(item: tuple[str, str]) -> tuple[int, ...]:
        number_match = re.match(r"^([1-7](?:\.\d+){0,2})\b", item[0])
        if not number_match:
            return (99,)
        return tuple(int(part) for part in number_match.group(1).split("."))

    selected.sort(key=sort_key)
    return selected
