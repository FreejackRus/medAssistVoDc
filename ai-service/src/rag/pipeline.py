from __future__ import annotations

import logging
import re
import threading
from typing import Generator, Iterable

from src.config import settings
from src.llm.client import OllamaClient
from src.llm.prompts import (
    ALGORITHM_SYSTEM,
    DIALOGUE_SYSTEM,
    EXPAND_SYSTEM,
    OUTLINE_SYSTEM,
    PHYSICIAN_ALGORITHM_SYSTEM,
    PHYSICIAN_EXPAND_SYSTEM,
    PHYSICIAN_OUTLINE_SYSTEM,
    STRUCTURED_ALGORITHM_SYSTEM,
    STRUCTURED_EXPAND_SYSTEM,
    STRUCTURED_OUTLINE_SYSTEM,
    algorithm_user_prompt,
    expand_section_prompt,
    outline_user_prompt,
    physician_algorithm_user_prompt,
    physician_outline_user_prompt,
    physician_section_prompt,
    structured_algorithm_user_prompt,
    structured_outline_user_prompt,
    structured_section_prompt,
)
from src.pdf.parser import (
    build_algorithm_sections,
    extract_definition,
    extract_diagnosis,
    parse_sections,
)
from src.rag.retriever import retrieve

log = logging.getLogger(__name__)

_llm: OllamaClient | None = None
_llm_lock = threading.Lock()

CHARS_PER_TOKEN_ESTIMATE = 2.5
ALGORITHM_PROMPT_OVERHEAD_TOKENS = 2500
ALGORITHM_OUTPUT_RESERVE_TOKENS = 4096
ALGORITHM_TEMPERATURE = 0.0
ALGORITHM_SEED = 42
MIN_DOCUMENT_BUDGET_TOKENS = 2048
ALGORITHM_CONTEXT_HARD_CHAR_CAP = settings.algorithm_context_char_cap
TRIMMED_TEXT_MARKER = (
    "\n\n[... ЧАСТЬ ТЕКСТА КЛИНИЧЕСКИХ РЕКОМЕНДАЦИЙ СОКРАЩЕНА ИЗ-ЗА ОГРАНИЧЕНИЯ КОНТЕКСТА ...]\n\n"
)
OUTLINE_SUMMARY_CHAR_CAP = 800
OUTLINE_FOR_EXPAND_CHAR_CAP = 2500
EXPAND_SECTION_CHAR_CAP = 6000

DEFAULT_CHAPTER_NAMES = {
    1: "Краткая информация по заболеванию или состоянию",
    2: "Диагностика заболевания или состояния",
    3: "Лечение заболевания или состояния",
    4: "Медицинская реабилитация и санаторно-курортное лечение",
    5: "Профилактика и диспансерное наблюдение",
    6: "Организация оказания медицинской помощи",
    7: "Дополнительная информация",
}

ALGORITHM_MODE_STRUCTURED = "structured"
ALGORITHM_MODE_SOURCE = "source"
ALGORITHM_MODE_PHYSICIAN = "physician"
ALGORITHM_MODES = frozenset(
    {
        ALGORITHM_MODE_STRUCTURED,
        ALGORITHM_MODE_SOURCE,
        ALGORITHM_MODE_PHYSICIAN,
    }
)
STRUCTURED_SOURCE_CHARS_PER_TOKEN_ESTIMATE = 1.8
STRUCTURED_SECTION_PROMPT_OVERHEAD_TOKENS = 3200
STRUCTURED_SUBSECTION_TRIM_MARKER = "\n\n[... пропущена часть подраздела ...]\n\n"
STRUCTURED_PRIORITY_EXCERPT_RATIO = 0.40
STRUCTURED_PRIORITY_EXCERPT_MARKER = (
    "\n\n[... дополнительные клинически значимые фрагменты того же подраздела ...]\n\n"
)
STRUCTURED_PRIORITY_MIN_SCORE = 7
STRUCTURED_PRIORITY_SENTENCE_MAX_CHARS = 1400
STRUCTURED_DEFINITION_DIRECT_CHAR_CAP = 2000
STRUCTURED_TERMS_DIRECT_CHAR_CAP = 2000
# Keep early clinical-assessment passes focused even when the model context is larger.
STRUCTURED_SECTION_BATCH_CHAR_CAP = 16000
STRUCTURED_BATCHED_SECTION_TITLES = frozenset(
    {
        "I. Предварительная оценка состояния пациента",
        "II. Этапы диагностики",
    }
)
STRUCTURED_ALGORITHM_SECTIONS = (
    (
        "I. Предварительная оценка состояния пациента",
        "Этиология и патогенез; классификация и пороговые статусы; причины и факторы "
        "риска; основные симптомы и признаки; состояния, требующие неотложной оценки. "
        "Определение уже выводится отдельно.",
        (1,),
    ),
    (
        "II. Этапы диагностики",
        "Первичное обследование; жалобы и анамнез; физикальное обследование; лабораторные "
        "и инструментальные исследования; критерии диагноза и интерпретация результатов.",
        (2,),
    ),
    (
        "III. Дифференциальная диагностика",
        "Заболевания и состояния для исключения; отличительные признаки; дополнительные "
        "исследования, прямо указанные для дифференциальной диагностики.",
        (2,),
    ),
    (
        "IV. Тактика лечения",
        "Немедикаментозная и медикаментозная терапия; процедуры и операции; показания, "
        "противопоказания, дозировки; контроль эффективности и безопасности.",
        (3,),
    ),
    (
        "V. Реабилитация и долгосрочное наблюдение",
        "Медицинская реабилитация; диспансерное и динамическое наблюдение; сроки контроля; "
        "профилактические меры и план наблюдения у специалистов.",
        (4, 5),
    ),
    (
        "VI. Особые ситуации",
        "Неотложные состояния и осложнения; критерии госпитализации и выписки; особые "
        "группы пациентов; организационные условия оказания помощи.",
        (3, 6, 7),
    ),
    (
        "VII. Дополнительно",
        "Факторы, влияющие на исход; критерии качества помощи; организация маршрутизации; "
        "дополнительные клинически значимые сведения, явно присутствующие в источнике.",
        (5, 6, 7),
    ),
)

PHYSICIAN_ALGORITHM_SECTIONS = (
    (
        "I. Паспорт заболевания",
        "Краткое определение; кодирование по МКБ; клиническая рекомендация или источник; "
        "классификация, формы, стадии, степени и диагностические пороги. Определение уже "
        "выводится отдельно.",
        (1,),
    ),
    (
        "II. Диагностический маршрут",
        "Жалобы и анамнез; физикальное обследование; лабораторные, инструментальные и "
        "иные исследования; консультации специалистов; критерии установления диагноза "
        "и дифференциальная диагностика.",
        (2,),
    ),
    (
        "III. Лечебная тактика",
        "Цели лечения; немедикаментозная и медикаментозная терапия; процедуры и операции; "
        "тактика по форме, стадии или тяжести; дозировки, сроки, противопоказания; контроль "
        "эффективности и безопасности.",
        (3,),
    ),
    (
        "IV. Маршрутизация и условия оказания помощи",
        "Амбулаторная и специализированная помощь; плановая и экстренная госпитализация; "
        "перевод; критерии выписки; организационные условия оказания помощи.",
        (6, 7),
    ),
    (
        "V. Наблюдение, реабилитация и профилактика",
        "Медицинская реабилитация; диспансерное и динамическое наблюдение; сроки и частота "
        "контроля; профилактические мероприятия и консультации специалистов.",
        (4, 5),
    ),
    (
        "VI. Чек-лист врача",
        "Проверяемые критерии качества помощи и обязательные действия из диагностики, "
        "лечения, маршрутизации и наблюдения. Условные действия сохраняют условие применения.",
        (7, 2, 3, 4, 5, 6),
    ),
)

STRUCTURED_SECTION_TITLE_HINTS = {
    "III. Дифференциальная диагностика": ("дифференц",),
    "VI. Особые ситуации": (
        "неотлож",
        "экстрен",
        "осложнен",
        "осложнён",
        "госпитал",
        "выписк",
        "опасно",
        "побочн",
        "особые групп",
        "острая ",
    ),
    "VII. Дополнительно": (
        "исход",
        "прогноз",
        "критерии качества",
        "маршрут",
        "организац",
        "профилактик",
        "диспансер",
        "госпитал",
        "выписк",
    ),
    "IV. Маршрутизация и условия оказания помощи": (
        "условия оказания",
        "маршрут",
        "организац",
        "госпитал",
        "стационар",
        "амбулатор",
        "перевод",
        "выписк",
        "неотлож",
        "экстрен",
    ),
}
STRUCTURED_SECTION_TITLE_EXCLUDES = {
    "II. Этапы диагностики": ("дифференц",),
}

NUMERIC_CITATION_PATTERN = re.compile(r"\s*\[(?:\s*\d+\s*(?:[,;\-–—]\s*\d+\s*)*)\]")
LOOSE_CITATION_END_PATTERN = re.compile(r"(?<![\w(])\d+(?:\s*[,;\-–—]\s*\d+)+\]")
INTERNAL_REFERENCE_PATTERN = re.compile(
    r"\s*\((?=[^()]{0,220}\b(?:табл(?:иц\w*)?\.?|рис(?:унк\w*)?\.?|"
    r"раздел\w*|приложени\w*)\s*(?:№\s*)?\d)[^()]{1,220}\)",
    flags=re.IGNORECASE,
)
DIRECT_INTERNAL_REFERENCE_PATTERN = re.compile(
    r"(?:\bсм\.\s*)?\b(?:табл(?:иц\w*)?\.?|рис(?:унк\w*)?\.?|раздел\w*|"
    r"приложени\w*)\s*(?:№\s*)?[А-ЯA-Z]?\d+(?:\.\d+)*",
    flags=re.IGNORECASE,
)
RELATIVE_INTERNAL_REFERENCE_PATTERN = re.compile(
    r"^\(\s*(?:см\.\s*)?(?:табл(?:иц\w*)?\.?|рис(?:унк\w*)?\.?|"
    r"раздел\w*|приложени\w*)\s+(?:выше|ниже|далее)[^)]*\)$",
    flags=re.IGNORECASE,
)
LATEX_COMMAND_REPLACEMENTS = {
    r"\geq": "≥",
    r"\ge": "≥",
    r"\leq": "≤",
    r"\le": "≤",
    r"\pm": "±",
    r"\times": "×",
    r"\rightarrow": "→",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\%": "%",
}
SUBSCRIPT_TRANSLATION = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
SUPERSCRIPT_TRANSLATION = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
INLINE_LATEX_PATTERN = re.compile(r"\$([^$\n]{1,300})\$")
TABLE_NUMBER_PATTERN = re.compile(r"(?im)^\s*Таблица\s+\d+(?:\.\d+)*[.:]?\s*")
TERMS_SECTION_HEADING_PATTERN = re.compile(r"(?im)^\s*Термины и определения\.?\s*$")
CHAPTER_ONE_HEADING_PATTERN = re.compile(r"(?im)^\s*1\.?\s+Краткая информация\b")
DEFINITION_START_PATTERN = re.compile(
    r"(?=(?P<definition>(?:[А-ЯЁ][а-яё]+|[A-Z][a-z]+|[A-ZА-ЯЁ]{2,})"
    r"[^.!?]{0,120}?(?:\s[—–-]\s|[—–-](?=[А-ЯЁа-яёA-Za-z]))\s*"
    r"(?:это\s+)?[^.!?]{0,100}\b"
    r"(?i:заболев\w*|состояни\w*|синдром\w*|нарушени\w*|порок\w*|"
    r"опухол\w*|поняти\w*|вызванн\w*|характериз\w*|обусловлен\w*)))"
)
DEFINED_AS_PATTERN = re.compile(
    r"(?<!не )\bопредел(?:ен|ена|ено|ены|яется)\w*\s+как\b",
    flags=re.IGNORECASE,
)
STRUCTURED_PRIORITY_SENTENCE_PATTERN = re.compile(
    r"\S.*?(?:[.!?](?=\s+(?:[А-ЯЁA-Z0-9•]|$))|\Z)",
    flags=re.DOTALL,
)
STRUCTURED_PRIORITY_ACTION_PATTERN = re.compile(
    r"рекоменду|следует|необходим|долж|показан|противопоказ|назнач|"
    r"начин|примен|выполн|провод|дости",
    flags=re.IGNORECASE,
)
STRUCTURED_PRIORITY_VALUE_PATTERN = re.compile(
    r"(?:[<>≤≥]|\d)|(?:мг|мкг|г\b|мл|ммоль|нг/мл|МЕ|%|"
    r"час|сут|дн(?:я|ей)|недел|месяц|год)",
    flags=re.IGNORECASE,
)
STRUCTURED_PRIORITY_DECISION_PATTERN = re.compile(
    r"(?:если|при\s|для\s+пациент|целев|критери|порог|доз|"
    r"эффектив|безопас|контрол|монотерап|комбинац)",
    flags=re.IGNORECASE,
)
STRUCTURED_PRIORITY_TARGET_PATTERN = re.compile(
    r"целев\w*.{0,100}(?:[<>≤≥]|(?:менее|более|не\s+менее)\s+\d)",
    flags=re.IGNORECASE,
)
STRUCTURED_PRIORITY_CONDITIONAL_PATTERN = re.compile(
    r"\bесли\b.{0,300}(?:лечен|терап|назнач|начин)",
    flags=re.IGNORECASE,
)

ATTACHMENT_TEXT_CHAR_CAP = 7000
ATTACHMENT_RETRIEVAL_TOP_K = 6
ATTACHMENT_CONTEXT_CHAR_CAP = 12000

CHAPTER_NAME_PREFIXES = (
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

SUSPICIOUS_DIAGNOSIS_FRAGMENTS = (
    "неизвест",
    "подозрен",
    "исследован",
    "провод",
    "рекоменду",
    "пациент",
    "является",
    "представляет",
    "составляющ",
    "гетерогенным",
)


def _resolve_source_diagnosis(source_text: str, diagnosis_name: str) -> str:
    inferred_name, _ = extract_diagnosis("", source_text, allow_llm_fallback=False)
    if not inferred_name:
        return diagnosis_name

    current = diagnosis_name.strip()
    if inferred_name.lower() != current.lower() and (
        not current
        or any(fragment in current.lower() for fragment in SUSPICIOUS_DIAGNOSIS_FRAGMENTS)
    ):
        log.info(
            "Using diagnosis inferred from source text for generation: stored=%r inferred=%r",
            current,
            inferred_name,
        )
        return inferred_name
    return diagnosis_name


def _looks_like_chapter_name(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", name.lower().replace("ё", "е")).strip()
    return any(normalized.startswith(prefix) for prefix in CHAPTER_NAME_PREFIXES)


def _get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = OllamaClient()
    return _llm


def _trim(
    text: str,
    *,
    max_tokens: int | None = None,
    reserve_tokens: int = 0,
) -> str:
    total_tokens = max_tokens or settings.max_context_tokens
    available_tokens = max(MIN_DOCUMENT_BUDGET_TOKENS, total_tokens - reserve_tokens)
    max_chars = int(available_tokens * CHARS_PER_TOKEN_ESTIMATE)

    if len(text) > max_chars:
        log.warning(
            "Prompt text trimmed to fit context: original_chars=%d trimmed_chars=%d total_tokens=%d reserve_tokens=%d",
            len(text),
            max_chars,
            total_tokens,
            reserve_tokens,
        )

        if max_chars <= len(TRIMMED_TEXT_MARKER) + 256:
            return text[:max_chars].rsplit("\n", 1)[0]

        content_chars = max_chars - len(TRIMMED_TEXT_MARKER)
        head_chars = int(content_chars * 0.7)
        tail_chars = content_chars - head_chars

        head = text[:head_chars].rsplit("\n", 1)[0].rstrip()
        tail = text[-tail_chars:].split("\n", 1)[-1].lstrip()
        return f"{head}{TRIMMED_TEXT_MARKER}{tail}"
    return text


def _truncate_block(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < 400:
        return text[:limit].rsplit("\n", 1)[0]

    marker = "\n\n[... раздел сокращен ...]"
    usable = max(0, limit - len(marker))
    head = int(usable * 0.8)
    tail = usable - head
    return (
        text[:head].rsplit("\n", 1)[0].rstrip() + marker + text[-tail:].split("\n", 1)[-1].lstrip()
    )


def _structured_section_source_char_cap() -> int:
    available_tokens = max(
        MIN_DOCUMENT_BUDGET_TOKENS,
        settings.max_context_tokens
        - STRUCTURED_SECTION_PROMPT_OVERHEAD_TOKENS
        - ALGORITHM_OUTPUT_RESERVE_TOKENS,
    )
    estimated_chars = int(available_tokens * STRUCTURED_SOURCE_CHARS_PER_TOKEN_ESTIMATE)
    return min(ALGORITHM_CONTEXT_HARD_CHAR_CAP, estimated_chars)


def _sanitize_structured_source_text(text: str) -> str:
    """Remove editorial references that are unusable outside the source document."""
    cleaned = NUMERIC_CITATION_PATTERN.sub("", text)
    cleaned = LOOSE_CITATION_END_PATTERN.sub("", cleaned)
    cleaned = INTERNAL_REFERENCE_PATTERN.sub("", cleaned)
    cleaned = TABLE_NUMBER_PATTERN.sub("Табличные данные: ", cleaned)
    cleaned = DIRECT_INTERNAL_REFERENCE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _sanitize_structured_output_fragment(fragment: str) -> str:
    if fragment.startswith("[") and NUMERIC_CITATION_PATTERN.fullmatch(fragment):
        return ""
    if fragment.startswith("(") and (
        INTERNAL_REFERENCE_PATTERN.fullmatch(fragment)
        or RELATIVE_INTERNAL_REFERENCE_PATTERN.fullmatch(fragment)
    ):
        return ""
    if fragment.startswith("$") and fragment.endswith("$"):
        return _normalize_inline_latex(fragment[1:-1])
    return INLINE_LATEX_PATTERN.sub(
        lambda match: _normalize_inline_latex(match.group(1)),
        fragment,
    )


def _normalize_inline_latex(value: str) -> str:
    normalized = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    for command, replacement in LATEX_COMMAND_REPLACEMENTS.items():
        normalized = normalized.replace(command, replacement)

    normalized = re.sub(r"_\{([^{}]+)\}|_([0-9])", _subscript_group, normalized)
    normalized = re.sub(r"\^\{([^{}]+)\}|\^([0-9])", _superscript_group, normalized)
    normalized = normalized.replace(r"\,", " ")
    normalized = re.sub(r"\\([A-Za-z]+)", r"\1", normalized)
    return normalized.replace("{", "").replace("}", "").strip()


def _subscript_group(match: re.Match[str]) -> str:
    value = match.group(1) or match.group(2)
    return value.translate(SUBSCRIPT_TRANSLATION)


def _superscript_group(match: re.Match[str]) -> str:
    value = match.group(1) or match.group(2)
    return value.translate(SUPERSCRIPT_TRANSLATION)


def _sanitize_structured_output_stream(
    tokens: Iterable[str],
) -> Generator[str, None, None]:
    buffered = ""
    closing = ""
    for token in tokens:
        emitted: list[str] = []
        plain: list[str] = []
        for char in token:
            if buffered:
                buffered += char
                if char == closing:
                    emitted.append(_sanitize_structured_output_fragment(buffered))
                    buffered = ""
                    closing = ""
                elif len(buffered) >= 300:
                    emitted.append(buffered)
                    buffered = ""
                    closing = ""
                continue

            if char in "([$":
                if plain:
                    emitted.append("".join(plain))
                    plain = []
                buffered = char
                closing = {"(": ")", "[": "]", "$": "$"}[char]
            else:
                plain.append(char)

        if plain:
            emitted.append("".join(plain))
        output = "".join(emitted)
        if output:
            yield output

    if buffered:
        yield buffered


def _section_heading_key(value: str) -> str:
    value = re.sub(r"^\s*#{1,6}\s*", "", value)
    value = re.sub(r"^\s*[IVXLCDM]+\.\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip(" .:").lower()


def _strip_repeated_section_heading_stream(
    tokens: Iterable[str],
    section_title: str,
) -> Generator[str, None, None]:
    expected = _section_heading_key(section_title)
    buffered = ""
    passthrough = False
    trim_leading_newlines = False

    for token in tokens:
        if passthrough:
            if trim_leading_newlines:
                token = token.lstrip("\r\n")
                if not token:
                    continue
                trim_leading_newlines = False
            yield token
            continue

        buffered += token
        candidate = buffered.lstrip()
        if not candidate:
            continue
        if not candidate.startswith("#"):
            yield buffered
            buffered = ""
            passthrough = True
            continue
        if "\n" not in candidate and len(candidate) < 300:
            continue

        first_line, separator, remainder = candidate.partition("\n")
        if _section_heading_key(first_line) == expected:
            cleaned_remainder = remainder.lstrip("\r\n") if separator else ""
            if cleaned_remainder:
                yield cleaned_remainder
            else:
                trim_leading_newlines = True
        else:
            yield buffered
        buffered = ""
        passthrough = True

    if buffered:
        candidate = buffered.lstrip()
        if _section_heading_key(candidate) != expected:
            yield buffered


def _prepare_structured_definition(definition: str) -> str:
    """Keep source definitions verbatim while omitting tables and editorial prose."""
    cleaned = _sanitize_structured_source_text(definition)
    if not cleaned:
        return ""
    if len(cleaned) <= STRUCTURED_DEFINITION_DIRECT_CHAR_CAP and "Табличные данные:" not in cleaned:
        return cleaned

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", normalized)
    selected: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        definition_starts = list(DEFINITION_START_PATTERN.finditer(sentence))
        if definition_starts:
            definition_start = min(
                definition_starts,
                key=lambda match: len(
                    re.split(
                        r"(?:\s[—–-]\s|[—–-](?=[А-ЯЁа-яёA-Za-z]))",
                        match.group("definition"),
                        maxsplit=1,
                    )[0].strip()
                ),
            )
            selected.append(sentence[definition_start.start("definition") :])
        elif (
            sentence
            and DEFINED_AS_PATTERN.search(sentence)
            and not re.match(r"^В\s+\d{4}\s+г\.", sentence)
        ):
            selected.append(sentence)

    if not selected and sentences:
        selected.append(sentences[0].strip())
    return "\n\n".join(dict.fromkeys(selected))


def _split_chapter_subsections(
    chapter_title: str,
    body: str,
) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^\[([^\]\n]+)\]\n", body))
    if not matches:
        return [(chapter_title, body.strip())]

    subsections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[match.end() : end].strip()
        if content:
            subsections.append((match.group(1).strip(), content))
    return subsections


def _extract_terms_section(full_text: str) -> str:
    for heading in reversed(list(TERMS_SECTION_HEADING_PATTERN.finditer(full_text))):
        chapter = CHAPTER_ONE_HEADING_PATTERN.search(full_text, heading.end())
        if not chapter:
            continue
        body = full_text[heading.end() : chapter.start()].strip()
        if len(body) >= 100:
            return body
    return ""


def _prepare_structured_terms(full_text: str) -> str:
    terms = _sanitize_structured_source_text(_extract_terms_section(full_text))
    terms = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", terms)
    terms = re.sub(r"\n{3,}", "\n\n", terms).strip()
    if not 100 <= len(terms) <= STRUCTURED_TERMS_DIRECT_CHAR_CAP:
        return ""
    return terms


def _with_structured_terms(
    chapters: list[tuple[int, str, str]],
    full_text: str,
) -> list[tuple[int, str, str]]:
    terms = _extract_terms_section(full_text)
    if not terms:
        return chapters
    return [
        (
            number,
            title,
            f"[Термины и определения]\n{terms}\n\n{body}" if number == 1 else body,
        )
        for number, title, body in chapters
    ]


def _with_structured_chapter_overviews(
    chapters: list[tuple[int, str, str]],
    full_text: str,
) -> list[tuple[int, str, str]]:
    """Restore facts located between a chapter heading and its first subsection."""
    overviews: dict[int, tuple[str, str]] = {}
    for title, body in parse_sections(full_text).items():
        match = re.match(r"^([1-7])\s+(.+)", title)
        if not match or not _looks_like_chapter_name(match.group(2)):
            continue
        content = body.strip()
        if content:
            overviews[int(match.group(1))] = (title, content)

    prepared: list[tuple[int, str, str]] = []
    for number, title, body in chapters:
        overview = overviews.get(number)
        if overview:
            overview_title, overview_body = overview
            body = f"[{overview_title} — вводная часть]\n{overview_body}\n\n{body}"
        prepared.append((number, title, body))
    return prepared


def _prepare_structured_chapters(
    chapters: list[tuple[int, str, str]],
    full_text: str,
) -> list[tuple[int, str, str]]:
    chapters_with_overviews = _with_structured_chapter_overviews(chapters, full_text)
    return _with_structured_terms(chapters_with_overviews, full_text)


def _prepare_physician_chapters(
    chapters: list[tuple[int, str, str]],
    full_text: str,
) -> list[tuple[int, str, str]]:
    return _with_structured_chapter_overviews(chapters, full_text)


def _select_structured_subsections(
    chapters: list[tuple[int, str, str]],
    section_title: str,
    source_numbers: tuple[int, ...],
) -> list[tuple[int, str, str]]:
    selected: list[tuple[int, str, str]] = []
    for number, chapter_title, body in chapters:
        if number not in source_numbers:
            continue
        selected.extend(
            (number, subsection_title, subsection_body)
            for subsection_title, subsection_body in _split_chapter_subsections(
                chapter_title,
                body,
            )
        )

    excludes = STRUCTURED_SECTION_TITLE_EXCLUDES.get(section_title, ())
    if excludes:
        selected = [
            subsection
            for subsection in selected
            if not any(hint in subsection[1].lower() for hint in excludes)
        ]

    hints = STRUCTURED_SECTION_TITLE_HINTS.get(section_title, ())
    if hints:
        focused = [
            subsection
            for subsection in selected
            if any(hint in subsection[1].lower() for hint in hints)
        ]
        if focused:
            selected = focused
    return selected


def _allocate_subsection_char_limits(
    lengths: list[int],
    budget: int,
    priority_indices: tuple[int, ...] = (),
) -> list[int]:
    if not lengths or budget <= 0:
        return [0] * len(lengths)
    if sum(lengths) <= budget:
        return lengths.copy()

    limits = [0] * len(lengths)
    remaining = budget
    unresolved = set(range(len(lengths)))
    for index in priority_indices:
        if index not in unresolved:
            continue
        reserved = min(lengths[index], remaining * 3 // 5, 10000)
        limits[index] = reserved
        remaining -= reserved
        unresolved.remove(index)

    while unresolved and remaining > 0:
        share = remaining // len(unresolved)
        if share <= 0:
            for index in sorted(unresolved)[:remaining]:
                limits[index] = 1
            break

        completed = [index for index in sorted(unresolved) if lengths[index] <= share]
        if not completed:
            for index in sorted(unresolved):
                limits[index] = share
            remainder = remaining - share * len(unresolved)
            for index in sorted(unresolved)[:remainder]:
                limits[index] += 1
            break

        for index in completed:
            limits[index] = lengths[index]
            remaining -= lengths[index]
            unresolved.remove(index)
    return limits


def _sample_subsection_overview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < 900 or limit >= int(len(text) * 0.65):
        return _truncate_block(text, limit)

    marker_chars = len(STRUCTURED_SUBSECTION_TRIM_MARKER) * 2
    usable = max(0, limit - marker_chars)
    head_chars = int(usable * 0.45)
    middle_chars = int(usable * 0.30)
    tail_chars = usable - head_chars - middle_chars

    head = text[:head_chars].rsplit("\n", 1)[0].rstrip()
    middle_start = max(head_chars, len(text) // 2 - middle_chars // 2)
    middle = text[middle_start : middle_start + middle_chars]
    middle = middle.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
    tail = text[-tail_chars:].split("\n", 1)[-1].lstrip()
    return STRUCTURED_SUBSECTION_TRIM_MARKER.join((head, middle, tail))


def _score_structured_priority_sentence(sentence: str) -> int:
    normalized = re.sub(r"\s+", " ", sentence).strip()
    if len(normalized) < 35:
        return -1

    has_action = STRUCTURED_PRIORITY_ACTION_PATTERN.search(normalized) is not None
    values = STRUCTURED_PRIORITY_VALUE_PATTERN.findall(normalized)
    decisions = STRUCTURED_PRIORITY_DECISION_PATTERN.findall(normalized)

    score = 5 if has_action else 0
    if values:
        score += 2 + min(3, len(values))
    score += min(4, len(decisions) * 2)
    if has_action and values:
        score += 3
    if values and STRUCTURED_PRIORITY_TARGET_PATTERN.search(normalized):
        score += 12
    if STRUCTURED_PRIORITY_CONDITIONAL_PATTERN.search(normalized):
        score += 12
    if len(normalized) > 1200:
        score -= 4
    return score


def _build_structured_priority_excerpt(text: str, budget: int) -> str:
    candidates: list[tuple[int, int, str]] = []
    max_sentence_chars = min(STRUCTURED_PRIORITY_SENTENCE_MAX_CHARS, budget)
    for order, match in enumerate(STRUCTURED_PRIORITY_SENTENCE_PATTERN.finditer(text)):
        sentence = match.group(0).strip()
        score = _score_structured_priority_sentence(sentence)
        if score >= STRUCTURED_PRIORITY_MIN_SCORE and len(sentence) <= max_sentence_chars:
            candidates.append((score, order, sentence))

    selected: list[tuple[int, int, str]] = []
    remaining = budget
    for score, order, sentence in sorted(candidates, key=lambda item: (-item[0], item[1])):
        separator_chars = 2 if selected else 0
        if len(sentence) + separator_chars > remaining:
            continue
        selected.append((score, order, sentence))
        remaining -= len(sentence) + separator_chars

    return "\n\n".join(sentence for _, _, sentence in sorted(selected, key=lambda item: item[1]))


def _has_structured_conditional_decision(excerpt: str) -> bool:
    normalized = re.sub(r"\s+", " ", excerpt)
    return STRUCTURED_PRIORITY_CONDITIONAL_PATTERN.search(normalized) is not None


def _sample_subsection(text: str, limit: int) -> str:
    if len(text) <= limit or limit < 1200:
        return _sample_subsection_overview(text, limit)

    usable = limit - len(STRUCTURED_PRIORITY_EXCERPT_MARKER)
    priority_budget = max(300, int(usable * STRUCTURED_PRIORITY_EXCERPT_RATIO))
    overview_budget = usable - priority_budget
    priority_excerpt = _build_structured_priority_excerpt(text, priority_budget)
    if not priority_excerpt:
        return _sample_subsection_overview(text, limit)

    overview = _sample_subsection_overview(text, overview_budget)
    return overview + STRUCTURED_PRIORITY_EXCERPT_MARKER + priority_excerpt


# --- Dynamic chapter grouping ---


def _group_numbered_chapters(full_text: str) -> list[tuple[int, str, str]]:
    """Parse source sections and retain their clinical-recommendation chapter numbers."""
    all_sections = parse_sections(full_text)
    algorithm_sections = build_algorithm_sections(full_text)

    if not algorithm_sections:
        return []

    # Collect top-level section names from parse_sections output
    chapter_names: dict[int, str] = {}
    for key in all_sections:
        m = re.match(r"^([1-7])\s+(.+)", key)
        if m and _looks_like_chapter_name(m.group(2)):
            chapter_names[int(m.group(1))] = m.group(2).strip()

    # Group algorithm sections by chapter number, keeping full body text
    chapters: dict[int, list[str]] = {}
    for title, body in algorithm_sections:
        m = re.match(r"^([1-7])", title)
        if not m:
            continue
        num = int(m.group(1))
        chapters.setdefault(num, []).append(f"[{title}]\n{body}")

    result = []
    for num in sorted(chapters):
        name = chapter_names.get(num) or DEFAULT_CHAPTER_NAMES.get(num, f"Раздел {num}")
        merged = "\n\n".join(chapters[num])
        result.append((num, name, merged))
    return result


def _group_by_chapter(full_text: str) -> list[tuple[str, str]]:
    """Return source chapters in the shape used by the original generation mode."""
    return [(title, body) for _, title, body in _group_numbered_chapters(full_text)]


# --- Two-pass algorithm generation ---


def _build_outline_input(chapters: list[tuple[str, str]]) -> str:
    """Build XML-tagged section summaries for the outline prompt."""
    blocks: list[str] = []
    for title, content in chapters:
        summary = _truncate_block(content, OUTLINE_SUMMARY_CHAR_CAP)
        blocks.append(
            "\n".join(
                [
                    "<section>",
                    f"<title>{title}</title>",
                    "<content>",
                    summary,
                    "</content>",
                    "</section>",
                ]
            )
        )
    return "\n\n".join(blocks)


def _structured_section_plan() -> str:
    return "\n".join(f"- {title}: {focus}" for title, focus, _ in STRUCTURED_ALGORITHM_SECTIONS)


def _physician_section_plan() -> str:
    return "\n".join(f"- {title}: {focus}" for title, focus, _ in PHYSICIAN_ALGORITHM_SECTIONS)


def _render_structured_section_input(
    selected: list[tuple[int, str, str]],
) -> str:
    if not selected:
        return ""

    prepared: list[tuple[str, str, str]] = []
    overhead = len(selected) + max(0, (len(selected) - 1) * 2)
    for number, title, body in selected:
        prefix = "\n".join(
            [
                f'<source_subsection chapter="{number}">',
                f"<title>{title}</title>",
                "<content>",
            ]
        )
        suffix = "\n</content>\n</source_subsection>"
        clean_body = _sanitize_structured_source_text(body)
        prepared.append((prefix, clean_body, suffix))
        overhead += len(prefix) + len(suffix)

    char_cap = _structured_section_source_char_cap()
    body_budget = max(0, char_cap - overhead)
    limits = _allocate_subsection_char_limits(
        [len(body) for _, body, _ in prepared],
        body_budget,
        tuple(
            index
            for index, (_, title, _) in enumerate(selected)
            if title.lower() == "термины и определения"
        ),
    )
    blocks: list[str] = []
    priority_excerpts: list[str] = []
    for (prefix, body, suffix), limit in zip(prepared, limits, strict=True):
        if limit <= 0:
            continue
        sampled = _sample_subsection(body, limit)
        overview, marker, priority_excerpt = sampled.partition(STRUCTURED_PRIORITY_EXCERPT_MARKER)
        if marker:
            sampled = overview
            priority_excerpts.append(priority_excerpt)
        blocks.append(f"{prefix}\n{sampled}{suffix}")

    if priority_excerpts:
        priority_excerpts.sort(
            key=lambda excerpt: not _has_structured_conditional_decision(excerpt),
        )
        checklist = "\n".join(
            [
                "<priority_excerpts>",
                *(f"<excerpt>{excerpt}</excerpt>" for excerpt in priority_excerpts),
                "</priority_excerpts>",
            ]
        )
        blocks.insert(0, checklist)
    return "\n\n".join(blocks)


def _build_structured_section_input(
    chapters: list[tuple[int, str, str]],
    section_title: str,
    source_numbers: tuple[int, ...],
) -> str:
    selected = _select_structured_subsections(
        chapters,
        section_title,
        source_numbers,
    )
    return _render_structured_section_input(selected)


def _partition_structured_subsections(
    selected: list[tuple[int, str, str]],
    char_cap: int,
) -> list[list[tuple[int, str, str]]]:
    batches: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    current_chars = 0

    for subsection in selected:
        subsection_chars = len(subsection[2])
        if current and current_chars + subsection_chars > char_cap:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(subsection)
        current_chars += subsection_chars

    if current:
        batches.append(current)
    return batches


def _build_structured_section_inputs(
    chapters: list[tuple[int, str, str]],
    section_title: str,
    source_numbers: tuple[int, ...],
) -> list[str]:
    selected = _select_structured_subsections(
        chapters,
        section_title,
        source_numbers,
    )
    if not selected:
        return []
    if section_title not in STRUCTURED_BATCHED_SECTION_TITLES:
        return [_render_structured_section_input(selected)]

    batches = _partition_structured_subsections(
        selected,
        STRUCTURED_SECTION_BATCH_CHAR_CAP,
    )
    return [_render_structured_section_input(batch) for batch in batches]


def _build_physician_section_inputs(
    chapters: list[tuple[int, str, str]],
    section_title: str,
    source_numbers: tuple[int, ...],
) -> list[str]:
    if section_title != "VI. Чек-лист врача":
        return _build_structured_section_inputs(chapters, section_title, source_numbers)

    selected = _select_structured_subsections(chapters, section_title, source_numbers)
    quality_subsections = [
        subsection
        for subsection in selected
        if any(
            marker in subsection[1].lower()
            for marker in ("критерии качества", "критерии оценки качества", "чек-лист")
        )
    ]
    if quality_subsections:
        selected = quality_subsections + [
            subsection for subsection in selected if subsection not in quality_subsections
        ]
    return [_render_structured_section_input(selected)] if selected else []


def _stream_outline(
    diagnosis_name: str,
    chapters: list[tuple[str, str]],
) -> Generator[str, None, None]:
    """Pass 1: generate structural outline from chapter summaries."""
    sections_text = _build_outline_input(chapters)
    messages = [
        {"role": "system", "content": OUTLINE_SYSTEM},
        {"role": "user", "content": outline_user_prompt(diagnosis_name, sections_text)},
    ]
    yield from _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=2048,
        seed=ALGORITHM_SEED,
    )


def _stream_expansion(
    diagnosis_name: str,
    outline_text: str,
    section_title: str,
    section_body: str,
) -> Generator[str, None, None]:
    """Pass 2: stream detailed expansion of a single chapter."""
    trimmed_outline = _truncate_block(outline_text.strip(), OUTLINE_FOR_EXPAND_CHAR_CAP)
    trimmed_body = _truncate_block(section_body, EXPAND_SECTION_CHAR_CAP)
    messages = [
        {"role": "system", "content": EXPAND_SYSTEM},
        {
            "role": "user",
            "content": expand_section_prompt(
                diagnosis_name,
                trimmed_outline,
                section_title,
                trimmed_body,
            ),
        },
    ]
    yield from _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
        seed=ALGORITHM_SEED,
    )


def _stream_structured_outline(
    diagnosis_name: str,
    chapters: list[tuple[int, str, str]],
) -> Generator[str, None, None]:
    source_sections = _build_outline_input(
        [(title, _sanitize_structured_source_text(body)) for _, title, body in chapters]
    )
    messages = [
        {"role": "system", "content": STRUCTURED_OUTLINE_SYSTEM},
        {
            "role": "user",
            "content": structured_outline_user_prompt(
                diagnosis_name,
                source_sections,
                _structured_section_plan(),
            ),
        },
    ]
    yield from _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=2048,
        seed=ALGORITHM_SEED,
    )


def _stream_structured_expansion(
    diagnosis_name: str,
    outline_text: str,
    section_title: str,
    section_focus: str,
    section_text: str,
    already_rendered: str = "",
) -> Generator[str, None, None]:
    if already_rendered:
        section_text = "\n".join(
            [
                "<already_rendered>",
                already_rendered,
                "</already_rendered>",
                section_text,
            ]
        )
    messages = [
        {"role": "system", "content": STRUCTURED_EXPAND_SYSTEM},
        {
            "role": "user",
            "content": structured_section_prompt(
                diagnosis_name,
                _truncate_block(outline_text.strip(), OUTLINE_FOR_EXPAND_CHAR_CAP),
                section_title,
                section_focus,
                section_text,
            ),
        },
    ]
    output_tokens = _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
        seed=ALGORITHM_SEED,
    )
    yield from _sanitize_structured_output_stream(
        _strip_repeated_section_heading_stream(
            output_tokens,
            section_title,
        )
    )


def _stream_physician_outline(
    diagnosis_name: str,
    chapters: list[tuple[int, str, str]],
) -> Generator[str, None, None]:
    source_sections = _build_outline_input(
        [(title, _sanitize_structured_source_text(body)) for _, title, body in chapters]
    )
    messages = [
        {"role": "system", "content": PHYSICIAN_OUTLINE_SYSTEM},
        {
            "role": "user",
            "content": physician_outline_user_prompt(
                diagnosis_name,
                source_sections,
                _physician_section_plan(),
            ),
        },
    ]
    yield from _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=2048,
        seed=ALGORITHM_SEED,
    )


def _stream_physician_expansion(
    diagnosis_name: str,
    outline_text: str,
    section_title: str,
    section_focus: str,
    section_text: str,
    already_rendered: str = "",
) -> Generator[str, None, None]:
    if already_rendered:
        section_text = "\n".join(
            [
                "<already_rendered>",
                already_rendered,
                "</already_rendered>",
                section_text,
            ]
        )
    messages = [
        {"role": "system", "content": PHYSICIAN_EXPAND_SYSTEM},
        {
            "role": "user",
            "content": physician_section_prompt(
                diagnosis_name,
                _truncate_block(outline_text.strip(), OUTLINE_FOR_EXPAND_CHAR_CAP),
                section_title,
                section_focus,
                section_text,
            ),
        },
    ]
    output_tokens = _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
        seed=ALGORITHM_SEED,
    )
    yield from _sanitize_structured_output_stream(
        _strip_repeated_section_heading_stream(
            output_tokens,
            section_title,
        )
    )


def stream_algorithm(
    full_text: str,
    diagnosis_name: str,
    mode: str = ALGORITHM_MODE_PHYSICIAN,
) -> Generator[str, None, None]:
    """Generate an algorithm in the selected stable output mode."""
    if mode not in ALGORITHM_MODES:
        raise ValueError(f"Unsupported algorithm generation mode: {mode}")

    diagnosis_name = _resolve_source_diagnosis(full_text, diagnosis_name)
    if mode == ALGORITHM_MODE_SOURCE:
        yield from _stream_source_sections_algorithm(full_text, diagnosis_name)
        return
    if mode == ALGORITHM_MODE_PHYSICIAN:
        yield from _stream_physician_algorithm(full_text, diagnosis_name)
        return

    yield from _stream_structured_algorithm(full_text, diagnosis_name)


def _stream_source_sections_algorithm(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Original two-pass mode that follows the source document chapter structure."""
    chapters = _group_by_chapter(full_text)

    if len(chapters) < 2:
        log.info("Too few source chapters (%d), falling back to single-pass", len(chapters))
        yield from _stream_algorithm_single_pass(full_text, diagnosis_name)
        return

    log.info(
        "Two-pass generation: %d chapters for '%s'",
        len(chapters),
        diagnosis_name,
    )

    # Pass 1: generate outline internally (not streamed to user)
    outline_parts: list[str] = []
    for token in _stream_outline(diagnosis_name, chapters):
        outline_parts.append(token)
    outline_text = "".join(outline_parts)
    log.info("Outline generated (%d chars), starting expansion", len(outline_text))

    # Pass 2: stream each chapter expansion token-by-token
    for i, (title, body) in enumerate(chapters):
        yield f"## {title}\n\n"
        for token in _stream_expansion(diagnosis_name, outline_text, title, body):
            yield token
        if i < len(chapters) - 1:
            yield "\n\n"

    log.info("Two-pass generation complete for '%s'", diagnosis_name)


def _stream_structured_algorithm(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Two-pass practical algorithm with a fixed clinician-oriented structure."""
    chapters = _prepare_structured_chapters(_group_numbered_chapters(full_text), full_text)
    if len(chapters) < 2:
        log.info(
            "Too few source chapters (%d), using structured single-pass",
            len(chapters),
        )
        yield from _stream_structured_algorithm_single_pass(full_text, diagnosis_name)
        return

    log.info(
        "Structured two-pass generation: %d chapters for '%s'",
        len(chapters),
        diagnosis_name,
    )

    outline_parts: list[str] = []
    for token in _stream_structured_outline(diagnosis_name, chapters):
        outline_parts.append(token)
    outline_text = "".join(outline_parts)
    exact_definition = _prepare_structured_definition(extract_definition(full_text))
    exact_terms = _prepare_structured_terms(full_text)

    yield f"# Расширенный алгоритм диагностики и лечения: {diagnosis_name}\n\n"
    for index, (title, focus, source_numbers) in enumerate(STRUCTURED_ALGORITHM_SECTIONS):
        yield f"## {title}\n\n"
        already_rendered_parts: list[str] = []
        if index == 0 and exact_definition:
            already_rendered_parts.append(exact_definition)
            yield "### Определение заболевания или состояния\n\n"
            yield exact_definition
            yield "\n\n"
        if index == 0 and exact_terms and exact_terms not in exact_definition:
            already_rendered_parts.append(exact_terms)
            yield "### Термины и определения\n\n"
            yield exact_terms
            yield "\n\n"
        already_rendered = "\n\n".join(already_rendered_parts)
        section_inputs = _build_structured_section_inputs(
            chapters,
            title,
            source_numbers,
        )
        if section_inputs:
            for part_index, section_text in enumerate(section_inputs):
                if part_index:
                    yield "\n\n"
                yield from _stream_structured_expansion(
                    diagnosis_name,
                    outline_text,
                    title,
                    focus,
                    section_text,
                    already_rendered if part_index == 0 else "",
                )
        else:
            yield "- В клинических рекомендациях отдельные сведения для этого раздела не приведены."
        if index < len(STRUCTURED_ALGORITHM_SECTIONS) - 1:
            yield "\n\n"

    log.info("Structured two-pass generation complete for '%s'", diagnosis_name)


def _stream_physician_algorithm(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Two-pass working algorithm modeled after physician-authored local checklists."""
    chapters = _prepare_physician_chapters(_group_numbered_chapters(full_text), full_text)
    if len(chapters) < 2:
        log.info(
            "Too few source chapters (%d), using physician single-pass",
            len(chapters),
        )
        yield from _stream_physician_algorithm_single_pass(full_text, diagnosis_name)
        return

    log.info(
        "Physician two-pass generation: %d chapters for '%s'",
        len(chapters),
        diagnosis_name,
    )

    outline_parts: list[str] = []
    for token in _stream_physician_outline(diagnosis_name, chapters):
        outline_parts.append(token)
    outline_text = "".join(outline_parts)
    exact_definition = _prepare_structured_definition(extract_definition(full_text))

    yield f"# Алгоритм оказания медицинской помощи: {diagnosis_name}\n\n"
    for index, (title, focus, source_numbers) in enumerate(PHYSICIAN_ALGORITHM_SECTIONS):
        yield f"## {title}\n\n"
        already_rendered_parts: list[str] = []
        if index == 0 and exact_definition:
            already_rendered_parts.append(exact_definition)
            yield "### Определение\n\n"
            yield exact_definition
            yield "\n\n"
        already_rendered = "\n\n".join(already_rendered_parts)
        section_inputs = _build_physician_section_inputs(
            chapters,
            title,
            source_numbers,
        )
        if section_inputs:
            for part_index, section_text in enumerate(section_inputs):
                if part_index:
                    yield "\n\n"
                yield from _stream_physician_expansion(
                    diagnosis_name,
                    outline_text,
                    title,
                    focus,
                    section_text,
                    already_rendered if part_index == 0 else "",
                )
        else:
            if title == "VI. Чек-лист врача":
                yield ("| № | Критерий | Да | Нет | Комментарий |\n|---:|---|:---:|:---:|---|")
            else:
                yield (
                    "- В клинических рекомендациях отдельные сведения для этого раздела "
                    "не приведены."
                )
        if index < len(PHYSICIAN_ALGORITHM_SECTIONS) - 1:
            yield "\n\n"

    log.info("Physician two-pass generation complete for '%s'", diagnosis_name)


def _stream_algorithm_single_pass(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Fallback: single-pass algorithm generation with trimmed full text."""
    content = _trim(
        full_text,
        reserve_tokens=ALGORITHM_PROMPT_OVERHEAD_TOKENS + ALGORITHM_OUTPUT_RESERVE_TOKENS,
    )
    messages = [
        {"role": "system", "content": ALGORITHM_SYSTEM},
        {"role": "user", "content": algorithm_user_prompt(diagnosis_name, content)},
    ]
    yield from _get_llm().stream_chat(
        messages,
        temperature=ALGORITHM_TEMPERATURE,
        num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
        seed=ALGORITHM_SEED,
    )


def _stream_structured_algorithm_single_pass(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Fallback for documents whose chapter structure cannot be recovered."""
    content = _trim(
        _sanitize_structured_source_text(full_text),
        reserve_tokens=ALGORITHM_PROMPT_OVERHEAD_TOKENS + ALGORITHM_OUTPUT_RESERVE_TOKENS,
    )
    exact_definition = _prepare_structured_definition(extract_definition(full_text))
    if exact_definition:
        content = "\n".join(
            [
                "<verbatim_definition>",
                exact_definition,
                "</verbatim_definition>",
                content,
            ]
        )
    messages = [
        {"role": "system", "content": STRUCTURED_ALGORITHM_SYSTEM},
        {
            "role": "user",
            "content": structured_algorithm_user_prompt(
                diagnosis_name,
                content,
                _structured_section_plan(),
            ),
        },
    ]
    yield from _sanitize_structured_output_stream(
        _get_llm().stream_chat(
            messages,
            temperature=ALGORITHM_TEMPERATURE,
            num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
            seed=ALGORITHM_SEED,
        )
    )


def _stream_physician_algorithm_single_pass(
    full_text: str,
    diagnosis_name: str,
) -> Generator[str, None, None]:
    """Fallback for physician mode when chapter structure cannot be recovered."""
    content = _trim(
        _sanitize_structured_source_text(full_text),
        reserve_tokens=ALGORITHM_PROMPT_OVERHEAD_TOKENS + ALGORITHM_OUTPUT_RESERVE_TOKENS,
    )
    exact_definition = _prepare_structured_definition(extract_definition(full_text))
    if exact_definition:
        content = "\n".join(
            [
                "<verbatim_definition>",
                exact_definition,
                "</verbatim_definition>",
                content,
            ]
        )
    messages = [
        {"role": "system", "content": PHYSICIAN_ALGORITHM_SYSTEM},
        {
            "role": "user",
            "content": physician_algorithm_user_prompt(
                diagnosis_name,
                content,
                _physician_section_plan(),
            ),
        },
    ]
    yield from _sanitize_structured_output_stream(
        _get_llm().stream_chat(
            messages,
            temperature=ALGORITHM_TEMPERATURE,
            num_predict=ALGORITHM_OUTPUT_RESERVE_TOKENS,
            seed=ALGORITHM_SEED,
        )
    )


# --- RAG Q&A mode: chat ---


def stream_rag_answer(
    question: str,
    document_id: str,
    diagnosis_name: str,
    history: list[dict],
    attachments: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Retrieve relevant chunks, build context, and stream LLM answer."""
    context = retrieve(question, document_id, top_k=settings.top_k)
    attachment_context = _build_attachment_context(question, attachments or [])

    messages = [{"role": "system", "content": DIALOGUE_SYSTEM}]

    context_msg = (
        f"Диагноз: {diagnosis_name}\n\nРелевантные фрагменты клинических рекомендаций:\n{context}"
    )
    messages.append({"role": "system", "content": context_msg})
    if attachment_context:
        messages.append({"role": "system", "content": attachment_context})

    for msg in history[-20:]:
        messages.append(msg)

    messages.append({"role": "user", "content": question})

    yield from _get_llm().stream_chat(messages, temperature=0.4, num_predict=4096)


def _build_attachment_context(question: str, attachments: list[dict]) -> str:
    if not attachments:
        return ""

    blocks: list[str] = []
    total_chars = 0
    for index, attachment in enumerate(attachments, 1):
        filename = str(attachment.get("filename") or f"Вложение {index}")
        text = str(attachment.get("text") or "").strip()
        attachment_document_id = attachment.get("document_id")

        if not text:
            continue

        if attachment_document_id:
            retrieved = retrieve(
                question,
                str(attachment_document_id),
                top_k=ATTACHMENT_RETRIEVAL_TOP_K,
            ).strip()
        else:
            retrieved = ""

        if len(text) <= ATTACHMENT_TEXT_CHAR_CAP:
            material = text
        elif retrieved:
            material = (
                text[:1800].rsplit("\n", 1)[0].strip()
                + "\n\n[... полный текст вложения сокращен; ниже релевантные фрагменты ...]\n\n"
                + retrieved
            )
        else:
            material = _truncate_block(text, ATTACHMENT_TEXT_CHAR_CAP)

        block = "\n".join(
            [
                "<attachment>",
                f"<filename>{filename}</filename>",
                "<content>",
                material,
                "</content>",
                "</attachment>",
            ]
        )
        if total_chars + len(block) > ATTACHMENT_CONTEXT_CHAR_CAP:
            break
        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return ""

    return """Материалы пользователя для проверки.
Это НЕ клинические рекомендации и НЕ нормативный источник. Используй эти материалы как объект анализа/сравнения с клиническими рекомендациями.
Если пользователь просит проверить протокол, оцени соответствие протокола релевантным фрагментам клинических рекомендаций, укажи найденные расхождения, недостающие данные и что выглядит корректным.

<user_attachments>
{attachments}
</user_attachments>""".format(attachments="\n\n".join(blocks))
