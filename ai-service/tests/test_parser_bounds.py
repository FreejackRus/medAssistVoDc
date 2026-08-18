from __future__ import annotations

from src.pdf.parser import _get_main_content_bounds, extract_definition


def test_main_content_uses_last_chapter_start_and_real_appendix_heading() -> None:
    text = "\n".join(
        [
            "Оглавление",
            "1. Краткая информация",
            "Приложение А1. Состав рабочей группы",
            "Предисловие",
            "1. Краткая информация",
            "Основной текст",
            "Ссылка на следующей строке:",
            "Приложение А3 [36].",
            "Продолжение основного текста",
            "Приложение А1. Состав рабочей группы",
            "Текст приложения",
        ]
    )

    start, end = _get_main_content_bounds(text)

    assert text[start:].startswith("1. Краткая информация\nОсновной текст")
    assert text[end:].startswith("Приложение А1. Состав рабочей группы")
    assert "Продолжение основного текста" in text[start:end]


def test_main_content_keeps_bare_appendix_reference_line() -> None:
    text = "\n".join(
        [
            "1. Краткая информация",
            "Основной текст",
            "3. Лечение",
            "Показания суммированы в таблице П17,",
            "Приложение А3.",
            "3.1 Показания к терапии",
            "Рекомендуется начать терапию.",
            "Приложение А1. Состав рабочей группы",
            "Текст приложения",
        ]
    )

    start, end = _get_main_content_bounds(text)

    assert text[start:].startswith("1. Краткая информация")
    assert "3.1 Показания к терапии" in text[start:end]
    assert text[end:].startswith("Приложение А1. Состав рабочей группы")


def test_main_content_accepts_pdf_spacing_inside_heading() -> None:
    text = "\n".join(
        [
            "Оглавление",
            "1. Краткая информация................12",
            "1.  Краткая  информация по заболеванию",
            "Основной текст",
            "Приложение А1. Состав рабочей группы",
        ]
    )

    start, end = _get_main_content_bounds(text)

    assert text[start:].startswith("1.  Краткая  информация")
    assert text[end:].startswith("Приложение А1.")


def test_main_content_stops_before_references() -> None:
    text = "\n".join(
        [
            "1. Краткая информация",
            "Основной текст",
            "Список литературы ",
            "[1] Источник",
            "Приложение А1. Состав рабочей группы",
        ]
    )

    start, end = _get_main_content_bounds(text)

    assert text[start:].startswith("1. Краткая информация")
    assert text[end:].startswith("Список литературы")


def test_extract_definition_preserves_source_wording() -> None:
    definition = (
        "Заболевание — это точное официальное определение.\n"
        "Вторая строка определения также должна сохраниться."
    )
    text = "\n".join(
        [
            "Оглавление",
            "1. Краткая информация........................................5",
            "1.1 Определение заболевания или состояния....................6",
            "1.2 Этиология и патогенез.....................................7",
            "1. Краткая информация",
            "1.1 Определение заболевания или состояния",
            definition,
            "1.2 Этиология и патогенез",
            "Следующий раздел.",
        ]
    )

    assert extract_definition(text) == definition
