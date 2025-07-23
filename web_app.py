# web_app.py
import asyncio
import json
import re
import textwrap
from pathlib import Path
from typing import AsyncGenerator

import markdown
import weasyprint
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

from bot import MedicalAssistant

app = FastAPI(title="Medical Assistant Web")
templates_dir = Path(__file__).parent / "templates"

assistant = MedicalAssistant()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((templates_dir / "index.html").read_text(encoding="utf-8"))


# ---------- SSE в формате таблицы ----------
class AsyncStreamWriter:
    def __init__(self, queue: asyncio.Queue[str]) -> None:
        self.queue = queue
        self._closed = False

    def write(self, text: str) -> None:
        if not self._closed:
            self.queue.put_nowait(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
        self.queue.put_nowait("")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def parse_markdown_to_rows(md: str) -> list[dict]:
    """Превращает Markdown-таблицу в список словарей."""
    # ищем таблицу: | раздел | подраздел | детали |
    table = re.findall(r'^\|(.+?)\|\s*$', md, flags=re.M)
    if not table:
        return []

    rows = []
    for line in table:
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) == 3 and parts[0] and not parts[0].startswith('--'):
            rows.append({
                "Раздел": parts[0],
                "Подраздел": parts[1],
                "Детали": parts[2]
            })
    return rows


@app.post("/generate_table")
async def generate_table(pdf: UploadFile = File(...)):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)
    pdf_path = tmp_dir / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    async def event_stream() -> AsyncGenerator[str, None]:
        sections = await asyncio.get_running_loop().run_in_executor(
            None, assistant.load_guidelines, str(pdf_path)
        )
        if not sections:
            yield f"data: {json.dumps({'error':'Не удалось прочитать PDF'})}\n\n"
            return

        # собираем весь Markdown
        full_md = "\n".join(sections.values())
        rows = parse_markdown_to_rows(full_md)

        # стримим по строке
        for row in rows:
            yield f"data: {json.dumps({'row': row})}\n\n"

        # финальный объект для PDF
        yield f"data: {json.dumps({'done': rows})}\n\n"
        pdf_path.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- PDF из таблицы ----------
@app.post("/download_table_pdf")
async def download_table_pdf(request: dict) -> Response:
    rows = request.get("rows", [])
    if not rows:
        return Response(b"", media_type="application/pdf")

    cols = rows[0].keys()
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{row.get(c,'')}</td>" for c in cols) + "</tr>"
        for row in rows
    )
    html = textwrap.dedent(
        f"""
        <html>
          <head>
            <meta charset="utf-8">
            <title>Рекомендации</title>
            <style>
              body {{font-family:DejaVu Sans,Arial,Helvetica,sans-serif;margin:40px;color:#111827;}}
              table {{border-collapse:collapse;width:100%;}}
              th,td {{border:1px solid #d1d5db;padding:8px;text-align:left;}}
              th {{background:#f3f4f6;color:#0c4a6e;}}
            </style>
          </head>
          <body>
            <h1>Клинические рекомендации</h1>
            <table>
              <thead><tr>{head}</tr></thead>
              <tbody>{body}</tbody>
            </table>
          </body>
        </html>
        """
    )
    pdf_bytes = weasyprint.HTML(string=html).write_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=algorithm.pdf"},
    )