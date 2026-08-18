from __future__ import annotations

import glob as _glob
import logging
import re
import tempfile
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import markdown as _md
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.config import settings

log = logging.getLogger(__name__)

_font_registered = False
_font_name = "Helvetica"
_font_bold_name = "Helvetica-Bold"
_page_margin_horizontal = 16 * mm
_page_margin_top = 20 * mm
_page_margin_bottom = 18 * mm

_color_primary = colors.HexColor("#173F4F")
_color_accent = colors.HexColor("#0F766E")
_color_text = colors.HexColor("#24313D")
_color_muted = colors.HexColor("#61717D")
_color_border = colors.HexColor("#C8D3D9")
_color_surface = colors.HexColor("#F3F7F8")
_color_surface_alt = colors.HexColor("#F8FAFB")


def _ensure_font() -> None:
    """Register a Cyrillic-capable font (cross-platform detection)."""
    global _font_registered, _font_name, _font_bold_name
    if _font_registered:
        return

    for pattern in settings.font_paths:
        paths = _glob.glob(pattern) if "*" in pattern else [pattern]
        for path in paths:
            if Path(path).is_file():
                try:
                    pdfmetrics.registerFont(TTFont("CustomFont", path))
                    _font_name = "CustomFont"
                    _font_bold_name = "CustomFont"

                    font_path = Path(path)
                    bold_candidates = [
                        font_path.with_name("DejaVuSans-Bold.ttf"),
                        font_path.with_name("Arial Bold.ttf"),
                        font_path.with_name("arialbd.ttf"),
                    ]
                    for bold_path in bold_candidates:
                        if not bold_path.is_file():
                            continue
                        try:
                            pdfmetrics.registerFont(TTFont("CustomFontBold", str(bold_path)))
                            _font_bold_name = "CustomFontBold"
                            break
                        except Exception:
                            continue

                    pdfmetrics.registerFontFamily(
                        "CustomFont",
                        normal="CustomFont",
                        bold=_font_bold_name,
                        italic="CustomFont",
                        boldItalic=_font_bold_name,
                    )
                    _font_registered = True
                    log.info("Registered font: %s", path)
                    return
                except Exception:
                    continue

    log.warning("No Cyrillic font found, falling back to Helvetica")
    _font_registered = True


def _tag_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _inline_markup(element: ElementTree.Element) -> str:
    parts = [escape(element.text or "")]

    for child in element:
        tag = _tag_name(child)
        if tag in {"ul", "ol"}:
            rendered = ""
        elif tag in {"strong", "b"}:
            rendered = f"<b>{_inline_markup(child)}</b>"
        elif tag in {"em", "i"}:
            rendered = f"<i>{_inline_markup(child)}</i>"
        elif tag == "u":
            rendered = f"<u>{_inline_markup(child)}</u>"
        elif tag == "br":
            rendered = "<br/>"
        else:
            rendered = _inline_markup(child)

        parts.append(rendered)
        parts.append(escape(child.tail or ""))

    return "".join(parts).strip()


def _plain_text(element: ElementTree.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def _algorithm_diagnosis(md_text: str) -> str | None:
    for line in md_text.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if not match:
            continue

        heading = re.sub(r"[*_`]", "", match.group(1)).strip()
        title_match = re.match(
            r"^(?:Клинический\s+)?Алгоритм"
            r"(?:\s+оказания\s+медицинской\s+помощи)?\s*:\s*(.+)$",
            heading,
            flags=re.IGNORECASE,
        )
        return title_match.group(1).strip() if title_match else heading

    return None


def algorithm_pdf_title(md_text: str) -> str:
    diagnosis = _algorithm_diagnosis(md_text)
    return f"Алгоритм: {diagnosis}" if diagnosis else "Клинический алгоритм"


def algorithm_pdf_filename(md_text: str) -> str:
    diagnosis = _algorithm_diagnosis(md_text) or "клинический"
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", diagnosis)
    safe_name = re.sub(r"\s+", "_", safe_name).strip(" ._")
    safe_name = re.sub(r"_+", "_", safe_name) or "клинический"
    safe_name = safe_name.encode("utf-8")[:180].decode("utf-8", errors="ignore").rstrip(" ._")
    return f"Алгоритм_{safe_name}.pdf"


def _normalize_markdown(md_text: str) -> str:
    """Repair common model formatting without changing clinical wording."""
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[А-Яа-яЁё])-\n(?=[А-Яа-яЁё])", "-", text)
    lines = text.split("\n")
    normalized: list[str] = []
    table_separator = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    top_level_list_item = re.compile(r"^\s{0,3}(?:[-*+]\s+|\d+[.)]\s+)")
    standalone_bold = re.compile(r"^\s*\*\*[^*]+\*\*\s*$")

    for index, line in enumerate(lines):
        stripped = line.strip()
        next_is_table_separator = (
            index + 1 < len(lines) and table_separator.match(lines[index + 1]) is not None
        )
        starts_table = "|" in stripped and next_is_table_separator
        starts_list = top_level_list_item.match(line) is not None

        if normalized and normalized[-1].strip() and (starts_table or starts_list):
            previous = normalized[-1]
            continues_list = starts_list and top_level_list_item.match(previous) is not None
            if starts_table or not continues_list:
                normalized.append("")

        normalized.append(line.rstrip())
        if standalone_bold.match(line) and index + 1 < len(lines) and lines[index + 1].strip():
            normalized.append("")

    return "\n".join(normalized).strip()


def _build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    common = {
        "fontName": _font_name,
        "textColor": _color_text,
        "wordWrap": "CJK",
    }

    return {
        "normal": ParagraphStyle(
            "AlgorithmNormal",
            parent=sample["Normal"],
            fontSize=9.8,
            leading=13.5,
            spaceAfter=6,
            **common,
        ),
        "eyebrow": ParagraphStyle(
            "AlgorithmEyebrow",
            parent=sample["Normal"],
            fontName=_font_bold_name,
            fontSize=8,
            leading=10,
            textColor=_color_accent,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "AlgorithmH1",
            parent=sample["Heading1"],
            fontName=_font_bold_name,
            fontSize=18,
            leading=22,
            textColor=_color_primary,
            spaceBefore=0,
            spaceAfter=7,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "AlgorithmH2",
            parent=sample["Heading2"],
            fontName=_font_bold_name,
            fontSize=12,
            leading=15,
            textColor=_color_primary,
            spaceBefore=0,
            spaceAfter=0,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "AlgorithmH3",
            parent=sample["Heading3"],
            fontName=_font_bold_name,
            fontSize=10.8,
            leading=14,
            textColor=_color_primary,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "label": ParagraphStyle(
            "AlgorithmLabel",
            parent=sample["Normal"],
            fontName=_font_bold_name,
            fontSize=9.8,
            leading=13,
            textColor=_color_primary,
            spaceBefore=5,
            spaceAfter=4,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "list": ParagraphStyle(
            "AlgorithmList",
            parent=sample["Normal"],
            fontSize=9.8,
            leading=13.5,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=2.5,
            **common,
        ),
        "quote": ParagraphStyle(
            "AlgorithmQuote",
            parent=sample["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=_color_text,
            fontName=_font_name,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "AlgorithmCode",
            parent=sample["Code"],
            fontName=_font_name,
            fontSize=8.5,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=6,
            backColor=colors.HexColor("#F1F5F9"),
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "AlgorithmTableCell",
            parent=sample["Normal"],
            fontName=_font_name,
            fontSize=8.1,
            leading=10.2,
            textColor=_color_text,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "AlgorithmTableHeader",
            parent=sample["Normal"],
            fontName=_font_bold_name,
            fontSize=8,
            leading=10,
            textColor=colors.white,
            wordWrap="CJK",
        ),
    }


def _column_widths(rows: list[list[ElementTree.Element]], available_width: float) -> list[float]:
    column_count = max(len(row) for row in rows)
    headers = [
        _plain_text(rows[0][index]).strip().lower() if index < len(rows[0]) else ""
        for index in range(column_count)
    ]

    if column_count == 2:
        return [available_width * 0.34, available_width * 0.66]

    if (
        column_count == 5
        and headers[0].replace(".", "") in {"№", "no"}
        and headers[2:4] == ["да", "нет"]
    ):
        fractions = [0.07, 0.41, 0.07, 0.07, 0.38]
        return [available_width * fraction for fraction in fractions]

    weights: list[float] = []

    for column_index in range(column_count):
        texts = [_plain_text(row[column_index]) for row in rows if column_index < len(row)]
        longest_word = max(
            (len(word) for text in texts for word in text.split()),
            default=1,
        )
        average_length = sum(len(text) for text in texts) / max(len(texts), 1)
        weights.append(float(min(max(longest_word + average_length / 8, 8), 28)))

    total = sum(weights)
    return [available_width * weight / total for weight in weights]


def _table_flowable(
    element: ElementTree.Element,
    styles: dict[str, ParagraphStyle],
    available_width: float,
) -> Table | None:
    source_rows: list[list[ElementTree.Element]] = []
    header_rows = 0

    for row_element in element.iter():
        if _tag_name(row_element) != "tr":
            continue
        cells = [child for child in row_element if _tag_name(child) in {"th", "td"}]
        if not cells:
            continue
        source_rows.append(cells)
        if len(source_rows) == header_rows + 1 and all(_tag_name(cell) == "th" for cell in cells):
            header_rows += 1

    if not source_rows:
        return None

    column_count = max(len(row) for row in source_rows)
    header_labels = [
        _plain_text(source_rows[0][index]).strip().lower() if index < len(source_rows[0]) else ""
        for index in range(column_count)
    ]
    data: list[list[Paragraph]] = []
    for row_index, source_row in enumerate(source_rows):
        row: list[Paragraph] = []
        for column_index in range(column_count):
            if column_index < len(source_row):
                text = _inline_markup(source_row[column_index]) or " "
            else:
                text = " "
            if (
                row_index >= header_rows
                and text == " "
                and header_labels[column_index] in {"да", "нет"}
            ):
                text = "□"
            style = styles["table_header"] if row_index < header_rows else styles["table"]
            row.append(Paragraph(text, style))
        data.append(row)

    commands: list[tuple] = [
        ("BOX", (0, 0), (-1, -1), 0.55, _color_border),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, _color_border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header_rows:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), _color_primary),
                ("VALIGN", (0, 0), (-1, header_rows - 1), "MIDDLE"),
                ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, _color_primary),
            ]
        )
    if len(data) > header_rows:
        commands.append(
            (
                "ROWBACKGROUNDS",
                (0, header_rows),
                (-1, -1),
                [colors.white, _color_surface_alt],
            )
        )

    for column_index, label in enumerate(header_labels):
        if label.replace(".", "") in {"№", "no", "да", "нет"}:
            commands.append(("ALIGN", (column_index, 0), (column_index, -1), "CENTER"))

    return Table(
        data,
        colWidths=_column_widths(source_rows, available_width),
        repeatRows=header_rows,
        splitByRow=1,
        splitInRow=1,
        hAlign="LEFT",
        spaceBefore=4,
        spaceAfter=8,
        style=TableStyle(commands),
    )


def _section_heading(text: str, style: ParagraphStyle, available_width: float) -> Table:
    heading = Table(
        [[Paragraph(text, style)]],
        colWidths=[available_width],
        hAlign="LEFT",
        spaceBefore=11,
        spaceAfter=7,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _color_surface),
                ("LINEBEFORE", (0, 0), (0, -1), 3, _color_accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )
    heading.keepWithNext = True
    return heading


def _callout(text: str, style: ParagraphStyle, available_width: float) -> Table:
    return Table(
        [[Paragraph(text, style)]],
        colWidths=[available_width],
        hAlign="LEFT",
        spaceBefore=5,
        spaceAfter=8,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF7F5")),
                ("LINEBEFORE", (0, 0), (0, -1), 3, _color_accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )


def _append_element(
    story: list,
    element: ElementTree.Element,
    styles: dict[str, ParagraphStyle],
    available_width: float,
    list_level: int = 0,
) -> None:
    tag = _tag_name(element)

    if tag == "h1":
        plain_text = _plain_text(element)
        title_match = re.match(
            r"^Алгоритм оказания медицинской помощи\s*:\s*(.+)$",
            plain_text,
            flags=re.IGNORECASE,
        )
        if title_match:
            story.append(Paragraph("КЛИНИЧЕСКИЙ АЛГОРИТМ", styles["eyebrow"]))
            story.append(Paragraph(escape(title_match.group(1)), styles["h1"]))
        elif plain_text:
            story.append(Paragraph(escape(plain_text), styles["h1"]))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.2,
                color=_color_accent,
                spaceBefore=1,
                spaceAfter=7,
            )
        )
        return

    if tag == "h2":
        text = _inline_markup(element)
        if text:
            story.append(_section_heading(text, styles["h2"], available_width))
        return

    if tag in {"h3", "h4", "h5", "h6"}:
        text = _inline_markup(element)
        if text:
            story.append(Paragraph(text, styles["h3"]))
        return

    if tag == "p":
        text = _inline_markup(element)
        if text:
            children = list(element)
            standalone_bold = (
                len(children) == 1
                and _tag_name(children[0]) in {"strong", "b"}
                and not (element.text or "").strip()
                and not (children[0].tail or "").strip()
            )
            style = styles["label"] if standalone_bold else styles["normal"]
            story.append(Paragraph(_inline_markup(children[0]) if standalone_bold else text, style))
        return

    if tag in {"ul", "ol"}:
        ordered = tag == "ol"
        for index, item in enumerate(
            (child for child in element if _tag_name(child) == "li"),
            start=1,
        ):
            marker = f"{index}." if ordered else "•"
            text = _inline_markup(item)
            if text:
                style = ParagraphStyle(
                    f"AlgorithmListLevel{list_level}",
                    parent=styles["list"],
                    leftIndent=14 + list_level * 11,
                )
                story.append(Paragraph(f"{marker} {text}", style))
            for child in item:
                if _tag_name(child) in {"ul", "ol"}:
                    _append_element(
                        story,
                        child,
                        styles,
                        available_width,
                        list_level + 1,
                    )
        return

    if tag == "table":
        table = _table_flowable(element, styles, available_width)
        if table is not None:
            story.append(table)
        return

    if tag == "blockquote":
        text = _inline_markup(element)
        if text:
            story.append(_callout(text, styles["quote"], available_width))
        return

    if tag == "pre":
        text = escape("".join(element.itertext())).replace("\n", "<br/>")
        if text:
            story.append(Paragraph(text, styles["code"]))
        return

    if tag == "hr":
        story.append(Spacer(1, 6))
        return

    for child in element:
        _append_element(story, child, styles, available_width, list_level)


def _markdown_story(md_text: str, available_width: float) -> list:
    _ensure_font()
    styles = _build_styles()
    html_body = _md.markdown(
        _normalize_markdown(md_text),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="xhtml",
    )

    try:
        root = ElementTree.fromstring(f"<root>{html_body.replace('&nbsp;', '&#160;')}</root>")
    except ElementTree.ParseError:
        log.exception("Could not parse generated Markdown as XHTML")
        return [
            Paragraph(
                escape(_normalize_markdown(md_text)).replace("\n", "<br/>"),
                styles["normal"],
            )
        ]

    story: list = []
    for element in root:
        _append_element(story, element, styles, available_width)
    return story


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(_color_border)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 13 * mm, A4[0] - doc.rightMargin, 13 * mm)
    canvas.setFont(_font_name, 7.5)
    canvas.setFillColor(_color_muted)
    canvas.drawString(doc.leftMargin, 8.5 * mm, "МедАссистент · клинический алгоритм")
    canvas.drawRightString(A4[0] - doc.rightMargin, 8.5 * mm, f"Стр. {doc.page}")
    canvas.restoreState()


def _draw_first_page(canvas, doc) -> None:
    _draw_footer(canvas, doc)


def _draw_later_pages(canvas, doc) -> None:
    _draw_footer(canvas, doc)
    canvas.saveState()
    canvas.setFont(_font_bold_name, 7.5)
    canvas.setFillColor(_color_muted)
    canvas.drawString(doc.leftMargin, A4[1] - 11 * mm, "КЛИНИЧЕСКИЙ АЛГОРИТМ")
    canvas.setStrokeColor(_color_border)
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, A4[1] - 13 * mm, A4[0] - doc.rightMargin, A4[1] - 13 * mm)
    canvas.restoreState()


def markdown_to_pdf(md_text: str) -> str:
    """Convert markdown to PDF, return path to the generated file."""
    _ensure_font()

    export_dir = Path(settings.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(export_dir))
    tmp_path = tmp.name
    tmp.close()

    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=A4,
        leftMargin=_page_margin_horizontal,
        rightMargin=_page_margin_horizontal,
        topMargin=_page_margin_top,
        bottomMargin=_page_margin_bottom,
        title=algorithm_pdf_title(md_text),
        author="МедАссистент",
        subject="Алгоритм оказания медицинской помощи",
    )
    doc.build(
        _markdown_story(md_text, doc.width),
        onFirstPage=_draw_first_page,
        onLaterPages=_draw_later_pages,
    )
    return tmp_path
