from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import AlgorithmGenerate
from src.rag import pipeline

EXACT_DEFINITION = (
    "Тестовое заболевание — состояние, определённое в клинической рекомендации "
    "без каких-либо пересказов."
)


class FakeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def stream_chat(self, messages: list[dict], **_: object):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Распредели факты" in prompt or "Распредели темы" in prompt:
            yield "## Внутренний план\n- Факт из источника"
        elif "VI. Чек-лист врача" in prompt:
            yield (
                "| № | Критерий | Да | Нет | Комментарий |\n"
                "|---:|---|:---:|:---:|---|\n"
                "| 1 | Выполнено обязательное исследование |  |  |  |"
            )
        else:
            yield "### Проверенный блок\n- Факт из источника"


@pytest.fixture
def algorithm_source(monkeypatch: pytest.MonkeyPatch) -> FakeLlm:
    fake_llm = FakeLlm()
    chapters = [
        (number, pipeline.DEFAULT_CHAPTER_NAMES[number], f"Источник раздела {number}")
        for number in range(1, 8)
    ]
    monkeypatch.setattr(pipeline, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "_resolve_source_diagnosis",
        lambda _source, diagnosis: diagnosis,
    )
    monkeypatch.setattr(pipeline, "_group_numbered_chapters", lambda _source: chapters)
    monkeypatch.setattr(pipeline, "extract_definition", lambda _source: EXACT_DEFINITION)
    return fake_llm


def test_algorithm_request_defaults_to_physician_mode() -> None:
    request = AlgorithmGenerate(full_text="x" * 100, diagnosis_name="Тестовый диагноз")

    assert request.mode == "physician"


def test_algorithm_request_accepts_supported_modes_and_rejects_unknown_mode() -> None:
    request = AlgorithmGenerate(
        full_text="x" * 100,
        diagnosis_name="Тестовый диагноз",
        mode="source",
    )

    assert request.mode == "source"
    physician_request = AlgorithmGenerate(
        full_text="x" * 100,
        diagnosis_name="Тестовый диагноз",
        mode="physician",
    )
    assert physician_request.mode == "physician"
    with pytest.raises(ValidationError):
        AlgorithmGenerate(
            full_text="x" * 100,
            diagnosis_name="Тестовый диагноз",
            mode="unknown",
        )


def test_physician_mode_is_default_and_emits_working_sections(
    algorithm_source: FakeLlm,
) -> None:
    result = "".join(pipeline.stream_algorithm("source", "Тестовый диагноз"))

    assert result.startswith("# Алгоритм оказания медицинской помощи: Тестовый диагноз")
    for title, _, _ in pipeline.PHYSICIAN_ALGORITHM_SECTIONS:
        assert f"## {title}" in result
    assert f"### Определение\n\n{EXACT_DEFINITION}" in result
    assert result.count(EXACT_DEFINITION) == 1
    assert "| № | Критерий | Да | Нет | Комментарий |" in result
    assert all("**Кому:**" not in prompt for prompt in algorithm_source.prompts)
    assert all("**Когда:**" not in prompt for prompt in algorithm_source.prompts)
    assert len(algorithm_source.prompts) == 1 + len(pipeline.PHYSICIAN_ALGORITHM_SECTIONS)


def test_source_mode_preserves_document_chapter_headings(
    algorithm_source: FakeLlm,
) -> None:
    result = "".join(pipeline.stream_algorithm("source", "Тестовый диагноз", mode="source"))

    assert not result.startswith("# Расширенный алгоритм")
    assert EXACT_DEFINITION not in result
    for title in pipeline.DEFAULT_CHAPTER_NAMES.values():
        assert f"## {title}" in result
    assert any("**Кому:**" in prompt for prompt in algorithm_source.prompts)
    assert len(algorithm_source.prompts) == 1 + len(pipeline.DEFAULT_CHAPTER_NAMES)


def test_physician_mode_emits_working_sections_and_checklist(
    algorithm_source: FakeLlm,
) -> None:
    result = "".join(pipeline.stream_algorithm("source", "Тестовый диагноз", mode="physician"))

    assert result.startswith("# Алгоритм оказания медицинской помощи: Тестовый диагноз")
    for title, _, _ in pipeline.PHYSICIAN_ALGORITHM_SECTIONS:
        assert f"## {title}" in result
    assert f"### Определение\n\n{EXACT_DEFINITION}" in result
    assert result.count(EXACT_DEFINITION) == 1
    assert "| № | Критерий | Да | Нет | Комментарий |" in result
    assert all("**Кому:**" not in prompt for prompt in algorithm_source.prompts)
    assert all("**Когда:**" not in prompt for prompt in algorithm_source.prompts)
    assert len(algorithm_source.prompts) == 1 + len(pipeline.PHYSICIAN_ALGORITHM_SECTIONS)


def test_verbatim_definition_is_injected_only_into_structured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLlm()
    monkeypatch.setattr(pipeline, "_get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        pipeline,
        "_resolve_source_diagnosis",
        lambda _source, diagnosis: diagnosis,
    )
    monkeypatch.setattr(pipeline, "_group_numbered_chapters", lambda _source: [])
    monkeypatch.setattr(pipeline, "extract_definition", lambda _source: EXACT_DEFINITION)

    list(pipeline.stream_algorithm("source", "Тестовый диагноз"))

    structured_prompt = fake_llm.prompts[-1]
    assert "<verbatim_definition>" in structured_prompt
    assert EXACT_DEFINITION in structured_prompt

    fake_llm.prompts.clear()
    list(pipeline.stream_algorithm("source", "Тестовый диагноз", mode="source"))

    source_prompt = fake_llm.prompts[-1]
    assert "<verbatim_definition>" not in source_prompt
    assert EXACT_DEFINITION not in source_prompt


def test_pipeline_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported algorithm generation mode"):
        list(pipeline.stream_algorithm("source", "Тестовый диагноз", mode="unknown"))


def test_structured_source_removes_editorial_references() -> None:
    source = (
        "Порог описан в документе (табл. 1) [116] "
        "(более подробно эта информация изложена в разделе 1.5).\n"
        "Таблица 1. Интерпретация показателей [42].\n"
        "Критерии приведены в Приложении А3.4."
    )

    result = pipeline._sanitize_structured_source_text(source)

    assert "[116]" not in result
    assert "[42]" not in result
    assert "табл. 1" not in result
    assert "разделе 1.5" not in result
    assert "Приложении А3.4" not in result
    assert "Табличные данные: Интерпретация показателей" in result


def test_structured_output_stream_removes_references_across_tokens() -> None:
    tokens = iter(
        [
            "Порог [",
            "116] равен 20 нг/мл (см. таб",
            "лицу выше). Значение (25 нг/мл) сохраняется.",
        ]
    )

    result = "".join(pipeline._sanitize_structured_output_stream(tokens))

    assert "[116]" not in result
    assert "см. таблицу выше" not in result
    assert "Порог  равен 20 нг/мл" in result
    assert "(25 нг/мл)" in result


def test_structured_output_stream_normalizes_inline_latex() -> None:
    tokens = iter(
        [
            r"ФВ $\ge 50\%$, индекс $VO_{2peak}$, шкала $\text{CHA}_2",
            r"\text{DS}_2\text{-VASc}$.",
        ]
    )

    result = "".join(pipeline._sanitize_structured_output_stream(tokens))

    assert result == "ФВ ≥ 50%, индекс VO₂peak, шкала CHA₂DS₂-VASc."


def test_structured_output_stream_normalizes_latex_inside_parentheses() -> None:
    tokens = iter([r"Уровень (витамин $\text{D}_3$) ≥20 нг/мл ($\ge$50 нмоль/л)."])

    result = "".join(pipeline._sanitize_structured_output_stream(tokens))

    assert result == "Уровень (витамин D₃) ≥20 нг/мл (≥50 нмоль/л)."


@pytest.mark.parametrize(
    ("tokens", "section_title", "expected"),
    [
        (
            ["### II. Диагностический ", "маршрут\n\n### Жалобы\n- Факт"],
            "II. Диагностический маршрут",
            "### Жалобы\n- Факт",
        ),
        (
            ["### Лечебная тактика\n", "\n- Факт"],
            "III. Лечебная тактика",
            "- Факт",
        ),
        (
            ["### Кодирование по МКБ\n\n- L00"],
            "I. Паспорт заболевания",
            "### Кодирование по МКБ\n\n- L00",
        ),
    ],
)
def test_physician_output_removes_only_repeated_section_heading(
    tokens: list[str],
    section_title: str,
    expected: str,
) -> None:
    result = "".join(pipeline._strip_repeated_section_heading_stream(iter(tokens), section_title))

    assert result == expected


def test_structured_definition_keeps_definitions_but_omits_raw_table() -> None:
    source = "\n".join(
        [
            "Недостаточность витамина D — состояние, характеризующееся снижением уровня.",
            "Концентрация характеризует статус витамина D (табл. 1) [116].",
            "Таблица 1. Интерпретация концентраций",
            "Статус витамина D",
            "Концентрация, нг/мл",
            "Дефицит",
            "<20 нг/мл",
            "Гипервитаминоз D — патологическое состояние, вызванное интоксикацией [42].",
            "Рахит — заболевание детей раннего возраста.",
            "Дополнительное редакционное пояснение. " * 80,
        ]
    )

    result = pipeline._prepare_structured_definition(source)

    assert "Недостаточность витамина D — состояние" in result
    assert "Гипервитаминоз D — патологическое состояние" in result
    assert "Рахит — заболевание детей раннего возраста." in result
    assert "Статус витамина D" not in result
    assert "[42]" not in result


def test_short_terms_are_kept_verbatim_without_page_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terms = (
        "Индекс массы тела – используется для диагностики ожирения.\n"
        "Морбидное ожирение – это ожирение с ИМТ ≥ 35 кг/м2 при осложнениях; "
        "ИМТ ≥ 40 кг/м2 вне зависимости от осложнений.\n7"
    )
    monkeypatch.setattr(pipeline, "_extract_terms_section", lambda _source: terms)

    result = pipeline._prepare_structured_terms("source")

    assert "Морбидное ожирение – это ожирение" in result
    assert not result.endswith("7")


def test_large_terms_section_is_left_for_model_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_extract_terms_section",
        lambda _source: "Термин – точное определение. " * 200,
    )

    assert pipeline._prepare_structured_terms("source") == ""


def test_structured_context_is_distributed_across_subsections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapters = [
        (
            3,
            pipeline.DEFAULT_CHAPTER_NAMES[3],
            "\n\n".join(
                [
                    "[3.1 Первый подраздел]\n" + "А" * 2400,
                    "[3.2 Короткий подраздел]\nКОРОТКИЙ ФАКТ",
                    "[3.3 Последний подраздел]\n" + "Б" * 2400,
                ]
            ),
        )
    ]
    monkeypatch.setattr(pipeline, "_structured_section_source_char_cap", lambda: 1800)

    result = pipeline._build_structured_section_input(
        chapters,
        "IV. Тактика лечения",
        (3,),
    )

    assert len(result) <= 1800
    assert "3.1 Первый подраздел" in result
    assert "КОРОТКИЙ ФАКТ" in result
    assert "3.3 Последний подраздел" in result


def test_large_assessment_section_is_split_by_subsection() -> None:
    chapters = [
        (
            1,
            pipeline.DEFAULT_CHAPTER_NAMES[1],
            "\n\n".join(
                [
                    "[1.1 Первый подраздел]\n" + "А" * 9000,
                    "[1.2 Второй подраздел]\n" + "Б" * 9000,
                    "[1.3 Третий подраздел]\n" + "В" * 9000,
                ]
            ),
        )
    ]

    inputs = pipeline._build_structured_section_inputs(
        chapters,
        "I. Предварительная оценка состояния пациента",
        (1,),
    )

    assert len(inputs) == 3
    combined = "\n".join(inputs)
    assert combined.count("1.1 Первый подраздел") == 1
    assert combined.count("1.2 Второй подраздел") == 1
    assert combined.count("1.3 Третий подраздел") == 1


def test_large_treatment_section_remains_single_pass() -> None:
    chapters = [
        (
            3,
            pipeline.DEFAULT_CHAPTER_NAMES[3],
            "\n\n".join(
                [
                    "[3.1 Первый подраздел]\n" + "А" * 9000,
                    "[3.2 Второй подраздел]\n" + "Б" * 9000,
                ]
            ),
        )
    ]

    inputs = pipeline._build_structured_section_inputs(
        chapters,
        "IV. Тактика лечения",
        (3,),
    )

    assert len(inputs) == 1


def test_physician_passport_remains_single_pass_for_consistent_summary() -> None:
    chapters = [
        (
            1,
            pipeline.DEFAULT_CHAPTER_NAMES[1],
            "\n\n".join(
                [
                    "[1.1 Определение]\n" + "А" * 9000,
                    "[1.2 Кодирование]\n" + "Б" * 9000,
                    "[1.3 Классификация]\n" + "В" * 9000,
                ]
            ),
        )
    ]

    inputs = pipeline._build_physician_section_inputs(
        chapters,
        "I. Паспорт заболевания",
        (1,),
    )

    assert len(inputs) == 1
    assert "1.1 Определение" in inputs[0]
    assert "1.2 Кодирование" in inputs[0]
    assert "1.3 Классификация" in inputs[0]


def test_structured_context_places_priority_excerpts_before_source_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "\n".join(
        [
            "А" * 2200 + ".",
            (
                "Если целевой уровень превышен менее чем на 1,0%, лечение можно "
                "начинать с монотерапии метформином."
            ),
            "Б" * 7000 + ".",
        ]
    )
    chapters = [(3, pipeline.DEFAULT_CHAPTER_NAMES[3], source)]
    monkeypatch.setattr(pipeline, "_structured_section_source_char_cap", lambda: 2400)

    result = pipeline._build_structured_section_input(
        chapters,
        "IV. Тактика лечения",
        (3,),
    )

    assert len(result) <= 2400
    assert result.index("<priority_excerpts>") < result.index("<source_subsection")
    assert "менее чем на 1,0%" in result


def test_priority_excerpt_preserves_source_order() -> None:
    source = " ".join(
        [
            "Рекомендуется контроль показателя через 3 месяца.",
            (
                "Если целевой уровень превышен менее чем на 1,0%, лечение можно "
                "начинать с монотерапии метформином."
            ),
        ]
    )

    result = pipeline._build_structured_priority_excerpt(source, 1000)

    assert result.index("Рекомендуется контроль") < result.index("Если целевой уровень")


def test_conditional_decision_detection_handles_pdf_line_breaks() -> None:
    excerpt = (
        "Если целевой уровень превышен менее чем на 1,0%, то лечение\nможно начинать с монотерапии."
    )

    assert pipeline._has_structured_conditional_decision(excerpt)


def test_structured_sampler_adds_fact_outside_fixed_overview_windows() -> None:
    source = "\n".join(
        [
            "Начальная рекомендация применяется всем пациентам.",
            "А" * 2200 + ".",
            (
                "Если HbA1c превышает индивидуальный целевой уровень менее чем на 1,0%, "
                "лечение можно начинать с монотерапии метформином."
            ),
            "Б" * 5200 + ".",
            "Сведения в середине подраздела.",
            "В" * 5200 + ".",
            "Заключительные сведения подраздела.",
        ]
    )

    result = pipeline._sample_subsection(source, 2000)

    assert len(result) <= 2000
    assert "менее чем на 1,0%" in result
    assert "монотерапии метформином" in result
    assert "дополнительные клинически значимые фрагменты" in result


def test_structured_sampler_keeps_short_subsection_unchanged() -> None:
    source = "Рекомендуется контроль через 3 месяца."

    assert pipeline._sample_subsection(source, 1200) == source


def test_terms_are_added_only_to_structured_chapters() -> None:
    chapters = [
        (
            1,
            pipeline.DEFAULT_CHAPTER_NAMES[1],
            "[1.1 Определение]\nОсновной текст",
        )
    ]
    source = "\n".join(
        [
            "Термины и определения",
            "Специальный термин — точное определение для алгоритма. " * 3,
            "1. Краткая информация по заболеванию",
            "1.1 Определение",
            "Основной текст",
        ]
    )

    structured = pipeline._with_structured_terms(chapters, source)
    physician = pipeline._prepare_physician_chapters(chapters, source)

    assert "Специальный термин" not in chapters[0][2]
    assert "Специальный термин" in structured[0][2]
    assert "Специальный термин" not in physician[0][2]


def test_chapter_overviews_are_added_only_to_structured_chapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chapters = [
        (
            2,
            pipeline.DEFAULT_CHAPTER_NAMES[2],
            "[2.1 Жалобы и анамнез]\nОсновной текст подраздела",
        )
    ]
    source = "\n".join(
        [
            "2 Диагностика заболевания или состояния, медицинские показания и противопоказания",
            "Диагностический порог из вводной части главы.",
            "2.1 Жалобы и анамнез",
            "Основной текст подраздела",
        ]
    )
    monkeypatch.setattr(
        pipeline,
        "parse_sections",
        lambda _source: {
            "2 Диагностика заболевания или состояния": (
                "Диагностический порог из вводной части главы."
            )
        },
    )

    structured = pipeline._prepare_structured_chapters(chapters, source)

    assert "Диагностический порог" not in chapters[0][2]
    assert "Диагностический порог" in structured[0][2]
    assert "Основной текст подраздела" in structured[0][2]


def test_structured_prompt_requires_complete_markdown_tables() -> None:
    prompt = pipeline.structured_section_prompt(
        "Тестовый диагноз",
        "План",
        "II. Этапы диагностики",
        "Критерии диагноза",
        "Табличные данные",
    )

    assert "Markdown-таблицу" in prompt
    assert "не теряй строки" in prompt
    assert "Не используй Markdown-таблицы" not in prompt
    assert "номера библиографических источников" in prompt
    assert "каждый блок <source_subsection>" in prompt
    assert "не нормализуй единицы измерения" in prompt
    assert "не выводи одинаковые подписи строк" in prompt


def test_physician_prompt_requires_unfilled_quality_checklist() -> None:
    prompt = pipeline.physician_section_prompt(
        "Тестовый диагноз",
        "План",
        "VI. Чек-лист врача",
        "Критерии качества помощи",
        "Критерии из источника",
    )

    assert "| № | Критерий | Да | Нет | Комментарий |" in prompt
    assert "оставляй ячейки `Да`, `Нет` и `Комментарий` пустыми" in prompt
    assert "сначала перенеси явные критерии качества помощи" in prompt
    assert "Не превращай необязательное или условное действие в универсальное" in prompt
    assert "Не добавляй локальные согласования" in prompt
    assert "Объединяй повторяющиеся классификации" in prompt
    assert "не представляй варианты как одну универсальную норму" in prompt
    assert "Не пиши, что" in prompt


def test_physician_checklist_prioritizes_quality_criteria_without_losing_actions() -> None:
    chapters = [
        (
            2,
            pipeline.DEFAULT_CHAPTER_NAMES[2],
            "[2.1 Лабораторная диагностика]\nВыполнить обязательное исследование.",
        ),
        (
            7,
            pipeline.DEFAULT_CHAPTER_NAMES[7],
            "[7.1 Критерии оценки качества медицинской помощи]\nКритерий качества из рекомендации.",
        ),
    ]

    inputs = pipeline._build_physician_section_inputs(
        chapters,
        "VI. Чек-лист врача",
        (7, 2, 3, 4, 5, 6),
    )

    assert len(inputs) == 1
    assert "Критерий качества из рекомендации" in inputs[0]
    assert "Выполнить обязательное исследование" in inputs[0]
    assert inputs[0].index("Критерий качества") < inputs[0].index("обязательное исследование")
