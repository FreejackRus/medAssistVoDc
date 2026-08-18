from pathlib import Path
from urllib.parse import unquote

import fitz
from fastapi.testclient import TestClient
from reportlab.platypus import Paragraph, Table

from src.main import app
from src.pdf import exporter

PHYSICIAN_MARKDOWN = """# Алгоритм оказания медицинской помощи: Тест

## I. Паспорт заболевания

- Определение с кириллицей и порогом ≥ 20 нг/мл.

## II. Диагностический маршрут

| Показатель | Значение | Действие |
|---|---:|---|
| Риск | < 10% | Наблюдение |
| Риск | ≥ 10% | Дополнительное исследование |

## VI. Чек-лист врача

| № | Критерий | Да | Нет | Комментарий |
|---:|---|:---:|:---:|---|
| 1 | Выполнено обязательное исследование |  |  |  |
"""

COMPACT_MODEL_MARKDOWN = """# Алгоритм оказания медицинской помощи: Тест

## I. Паспорт заболевания

**Кодирование заболевания**
| Система | Код |
| :--- | :--- |
| МКБ-10 | E53.8 |

**Цели лечения**
* Предотвращение декомпенсации.
* Сохранение аутосомно-
рецессивного описания без лишнего пробела.

**Лечение метаболического криза**
Требуется экстренная госпитализация.
"""


def test_markdown_to_pdf_handles_physician_algorithm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(exporter.settings, "export_dir", str(tmp_path))

    output_path = Path(exporter.markdown_to_pdf(PHYSICIAN_MARKDOWN))
    pdf = output_path.read_bytes()

    assert output_path.parent == tmp_path
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000
    with fitz.open(output_path) as document:
        assert document.metadata["title"] == "Алгоритм: Тест"


def test_algorithm_pdf_filename_uses_sanitized_diagnosis() -> None:
    markdown = "# Алгоритм оказания медицинской помощи: Болезнь Фабри / классическая: форма?"

    assert (
        exporter.algorithm_pdf_filename(markdown) == "Алгоритм_Болезнь_Фабри_классическая_форма.pdf"
    )


def test_markdown_story_preserves_tables() -> None:
    story = exporter._markdown_story(PHYSICIAN_MARKDOWN, 500)

    tables = [flowable for flowable in story if isinstance(flowable, Table) and flowable._ncols > 1]

    assert len(tables) == 2
    assert tables[0]._ncols == 3
    assert tables[1]._ncols == 5


def test_markdown_story_repairs_compact_model_markdown() -> None:
    story = exporter._markdown_story(COMPACT_MODEL_MARKDOWN, 500)

    tables = [flowable for flowable in story if isinstance(flowable, Table)]
    paragraphs = [flowable.getPlainText() for flowable in story if isinstance(flowable, Paragraph)]

    assert len(tables) == 2  # Section band and Markdown table.
    assert tables[1]._ncols == 2
    assert "Цели лечения" in paragraphs
    assert "Лечение метаболического криза" in paragraphs
    assert "Требуется экстренная госпитализация." in paragraphs
    assert any("аутосомно-рецессивного" in text for text in paragraphs)


def test_exported_pdf_does_not_contain_raw_markdown_table(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(exporter.settings, "export_dir", str(tmp_path))

    output_path = exporter.markdown_to_pdf(COMPACT_MODEL_MARKDOWN)
    with fitz.open(output_path) as document:
        text = "\n".join(page.get_text() for page in document)

    assert "| :--- |" not in text
    assert "| Система | Код |" not in text
    assert "аутосомно-рецессивного" in text


def test_export_pdf_endpoint_returns_downloadable_pdf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(exporter.settings, "export_dir", str(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/algorithms/export-pdf",
            json={"markdown": PHYSICIAN_MARKDOWN},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment;" in response.headers["content-disposition"]
    assert unquote(response.headers["content-disposition"]).endswith("Алгоритм_Тест.pdf")
    assert response.content.startswith(b"%PDF-")
